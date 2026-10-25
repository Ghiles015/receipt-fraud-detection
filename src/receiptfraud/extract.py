"""Extraction des champs company / date / address / total.

Deux extracteurs :
- `RuleExtractor` : baseline sans apprentissage (regex + heuristiques de position).
- `LineClassifierExtractor` : classifieur par ligne (texte + mise en page) puis
  décodage par champ. Pour chaque champ :
    1. régression logistique sur n-grammes de caractères (ligne + contexte de rangée),
       prédictions out-of-fold pour éviter la fuite vers l'étage suivant ;
    2. gradient boosting (HistGradientBoosting) sur ~60 features de mise en page + le score texte.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from . import FIELDS
from .data import Document
from .features import line_features, row_context, _KW
from .labeling import line_labels
from .textutils import clean_company, find_amounts, find_dates, norm


@dataclass
class Extraction:
    values: dict = field(default_factory=dict)       # champ -> chaîne extraite
    confidence: dict = field(default_factory=dict)   # champ -> score [0, 1]
    lines: dict = field(default_factory=dict)        # champ -> indices des lignes sources
    line_probs: dict = field(default_factory=dict)   # champ -> probas par ligne (debug / démo)


def _text_inputs(doc: Document) -> list[str]:
    ctx = row_context(doc)
    return [f"{l.text} || {c}" for l, c in zip(doc.lines, ctx)]


# --------------------------------------------------------------------------- décodage

def decode(doc: Document, probs: dict[str, np.ndarray], thr: dict[str, float] | None = None) -> Extraction:
    thr = thr or {"company": 0.5, "address": 0.5}
    ex = Extraction(line_probs={k: v.tolist() for k, v in probs.items()})
    n = len(doc.lines)
    if n == 0:
        for f in FIELDS:
            ex.values[f], ex.confidence[f], ex.lines[f] = "", 0.0, []
        return ex

    # company : ligne la plus probable + voisines contiguës au-dessus du seuil
    p = probs["company"]
    i = int(np.argmax(p))
    span = [i]
    for step in (-1, 1):
        j = i + step
        while 0 <= j < n and p[j] >= thr["company"] and len(span) < 3:
            span.append(j)
            j += step
    span.sort()
    ex.values["company"] = clean_company(" ".join(doc.lines[k].text for k in span))
    ex.confidence["company"] = float(p[i])
    ex.lines["company"] = span

    # address : lignes au-dessus du seuil (hors company), plus grand bloc quasi contigu
    p = probs["address"].copy()
    p[span] = 0
    sel = [k for k in range(n) if p[k] >= thr["address"]]
    if not sel and p.max() > 0.2:
        sel = [int(np.argmax(p))]
    if sel:
        groups, cur = [], [sel[0]]
        for k in sel[1:]:
            if k - cur[-1] <= 2:
                cur.append(k)
            else:
                groups.append(cur)
                cur = [k]
        groups.append(cur)
        sel = max(groups, key=lambda g: sum(p[k] for k in g))
    ex.values["address"] = norm(" ".join(doc.lines[k].text for k in sel))
    ex.confidence["address"] = float(np.mean([p[k] for k in sel])) if sel else 0.0
    ex.lines["address"] = sel

    # date : meilleure ligne contenant une date
    p = probs["date"]
    cands = [(p[k], k, s) for k in range(n) for s, _ in find_dates(doc.lines[k].text)]
    if cands:
        sc, k, s = max(cands, key=lambda c: c[0])
        ex.values["date"], ex.confidence["date"], ex.lines["date"] = s, float(sc), [k]
    else:
        ex.values["date"], ex.confidence["date"], ex.lines["date"] = "", 0.0, []

    # total : meilleur montant (le plus à droite de la meilleure ligne)
    p = probs["total"]
    cands = [(p[k], k, find_amounts(doc.lines[k].text)[-1][1]) for k in range(n) if find_amounts(doc.lines[k].text)]
    if cands:
        sc, k, v = max(cands, key=lambda c: c[0])
        ex.values["total"], ex.confidence["total"], ex.lines["total"] = f"{v:.2f}", float(sc), [k]
    else:
        ex.values["total"], ex.confidence["total"], ex.lines["total"] = "", 0.0, []
    return ex


# --------------------------------------------------------------------------- baseline règles

class RuleExtractor:
    """Baseline déterministe : ce qu'on écrirait sans données d'entraînement."""

    def predict_probs(self, doc: Document) -> dict[str, np.ndarray]:
        n = len(doc.lines)
        probs = {f: np.zeros(n) for f in FIELDS}
        if n == 0:
            return probs
        texts = [l.text for l in doc.lines]
        first_date = next((i for i, t in enumerate(texts) if find_dates(t)), n)
        # company : 1re ligne avec un suffixe d'entreprise dans l'en-tête, sinon 1re ligne
        comp = next((i for i in range(min(first_date, 8)) if _KW["kw_sdn"].search(texts[i])), 0)
        probs["company"][comp] = 1.0
        # address : lignes d'en-tête après company avec mot-clé d'adresse, avant tél./GST
        for i in range(comp + 1, min(first_date, comp + 7)):
            if _KW["kw_tel"].search(texts[i]):
                break
            if _KW["kw_addr"].search(texts[i]):
                probs["address"][i] = 1.0
        # date : première date valide
        for i, t in enumerate(texts):
            if any(ymd for _, ymd in find_dates(t)):
                probs["date"][i] = 1.0
                break
        # total : montant sur la rangée d'un mot-clé TOTAL (hors sous-total), sinon le plus grand
        ctx = row_context(doc)
        best = None
        for i, t in enumerate(texts):
            if not find_amounts(t):
                continue
            s = t + " " + ctx[i]
            if _KW["kw_total"].search(s) and not _KW["kw_subtotal"].search(s) and not _KW["kw_tax"].search(s):
                best = i  # on garde la dernière occurrence
        if best is None:
            amts = [(find_amounts(t)[-1][1], i) for i, t in enumerate(texts) if find_amounts(t)]
            best = max(amts)[1] if amts else None
        if best is not None:
            probs["total"][best] = 1.0
        return probs

    def extract(self, doc: Document) -> Extraction:
        return decode(doc, self.predict_probs(doc))


# --------------------------------------------------------------------------- modèle appris

class LineClassifierExtractor:
    def __init__(self, n_folds: int = 5, seed: int = 0):
        self.n_folds = n_folds
        self.seed = seed
        self.vec = None
        self.text_models: dict = {}
        self.gbms: dict = {}
        self.thr = {"company": 0.5, "address": 0.5}

    def _make_vec(self):
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=40000,
                               sublinear_tf=True, lowercase=True)

    @staticmethod
    def _stack(docs):
        X, T, Y, G = [], [], {f: [] for f in FIELDS}, []
        for gi, d in enumerate(docs):
            if not d.lines:
                continue
            X.append(line_features(d))
            T.extend(_text_inputs(d))
            lab = line_labels(d)
            for f in FIELDS:
                Y[f].extend(lab[f])
            G.extend([gi] * len(d.lines))
        return np.vstack(X), T, {f: np.array(v) for f, v in Y.items()}, np.array(G)

    def fit(self, docs: list[Document]):
        X, T, Y, G = self._stack(docs)
        # 1) scores texte out-of-fold
        oof = np.zeros((len(T), len(FIELDS)))
        for tr, va in GroupKFold(self.n_folds).split(X, groups=G):
            vec = self._make_vec().fit([T[i] for i in tr])
            Xt_tr, Xt_va = vec.transform([T[i] for i in tr]), vec.transform([T[i] for i in va])
            for j, f in enumerate(FIELDS):
                lr = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
                lr.fit(Xt_tr, Y[f][tr])
                oof[va, j] = lr.predict_proba(Xt_va)[:, 1]
        self.vec = self._make_vec().fit(T)
        Xt = self.vec.transform(T)
        for f in FIELDS:
            self.text_models[f] = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(Xt, Y[f])
        # 2) gradient boosting sur features de mise en page + scores texte
        Xs = np.hstack([X, oof])
        for f in FIELDS:
            self.gbms[f] = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                random_state=self.seed).fit(Xs, Y[f])
        return self

    def predict_probs(self, doc: Document) -> dict[str, np.ndarray]:
        if not doc.lines:
            return {f: np.zeros(0) for f in FIELDS}
        X = line_features(doc)
        Xt = self.vec.transform(_text_inputs(doc))
        txt = np.column_stack([self.text_models[f].predict_proba(Xt)[:, 1] for f in FIELDS])
        Xs = np.hstack([X, txt])
        return {f: self.gbms[f].predict_proba(Xs)[:, 1] for f in FIELDS}

    def extract(self, doc: Document) -> Extraction:
        return decode(doc, self.predict_probs(doc), self.thr)
