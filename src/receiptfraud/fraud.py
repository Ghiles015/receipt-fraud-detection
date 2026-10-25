"""Détection d'incohérences et score de risque.

Trois couches :
1. `parse_financials` : reconstitue la structure financière du reçu
   (articles, sous-total, taxe, arrondi, total, espèces, rendu).
2. `run_checks` : règles métier explicables (chaque règle produit un indicateur et un écart).
3. `RiskModel` : combine les règles et un score d'anomalie (IsolationForest appris sur des
   reçus authentiques) en une probabilité de fraude calibrée, exprimée de 0 à 100.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import Document
from .extract import Extraction
from .features import _KW
from .textutils import find_amounts, find_dates

TOL = 0.06  # tolérance monétaire (arrondi à 5 sen en Malaisie + erreurs de 1 centime)

# Motifs plus stricts que ceux des features : « SR »/« ZR » sont des codes TVA d'articles,
# « GST REG NO » un identifiant, « TOTAL ... EXCL GST » un sous-total.
TAX_RE = re.compile(r"\b(GST|SST|TAX|SERVICE\s*CHARGE|SERV\.?\s*CH\w*|SVC\s*CHG|S\.?\s*CHARGE)\b", re.I)
DISCOUNT_RE = re.compile(r"\b(DISCOUNT|DISC|LESS|VOUCHER|DEPOSIT|PROMO|MEMBER\s*SAVING)\b", re.I)
NOT_TAX_RE = re.compile(r"(REG|ID\b|NO\b|NO\.|INCL|INCLUSIVE|INC\b|EXCL|SUMMARY|ITEM|AMOUNT\s*\(|SUPPLIES)", re.I)
SUBTOTAL_RE = re.compile(r"\b(SUB\s*-?\s*TOTAL|SUBTOTAL|NET\s*TOTAL)\b|EXCL|BEFORE\s*(GST|TAX)|W/O\s*GST", re.I)
NOT_FINAL_TOTAL_RE = re.compile(r"\b(QTY|QUANTITY|ITEMS?|DISCOUNT|DISC|SAVING|SAVINGS|POINTS?|WEIGHT|UNITS?|PCS|SUPPLIES|EXCL\w*)\b", re.I)


@dataclass
class Financials:
    total: float | None = None
    subtotal: float | None = None
    tax: float | None = None
    rounding: float | None = None
    cash: float | None = None
    change: float | None = None
    items: list[float] = field(default_factory=list)
    labelled_totals: list[float] = field(default_factory=list)
    discount: float = 0.0
    amounts: list[float] = field(default_factory=list)


def _row_labels(doc: Document) -> list[str]:
    """Libellé d'un montant = segments textuels à sa gauche sur la même rangée
    (les segments purement numériques, ex. une autre colonne de montants, sont ignorés)."""
    out = []
    for i, l in enumerate(doc.lines):
        left = [o for j, o in enumerate(doc.lines) if j != i and o.x1 <= l.x0 + 5
                and abs(o.cy - l.cy) < 0.6 * max(o.h, l.h) and re.search(r"[A-Za-z]{2}", o.text)]
        out.append(l.text + " " + " ".join(o.text for o in sorted(left, key=lambda o: o.x0)))
    return out


def parse_financials(doc: Document, ex: Extraction) -> Financials:
    fin = Financials()
    rows = _row_labels(doc)
    W = max(doc.width, 1.0)
    tv = ex.values.get("total")
    fin.total = float(tv) if tv else None
    total_line = ex.lines.get("total", [None])[0] if ex.lines.get("total") else None
    first_total_kw = None
    taxes = []
    amount_lines = [j for j, o in enumerate(doc.lines) if find_amounts(o.text)]

    def rightmost_in_row(i):
        l = doc.lines[i]
        return not any(j != i and doc.lines[j].x0 >= l.x1 - 5 and abs(doc.lines[j].cy - l.cy) < 0.6 * max(l.h, doc.lines[j].h)
                       for j in amount_lines)

    for i, l in enumerate(doc.lines):
        am = find_amounts(l.text)
        s = rows[i]
        if first_total_kw is None and _KW["kw_total"].search(s):
            first_total_kw = i
        if not am:
            continue
        v = am[-1][1]
        fin.amounts.append(v)
        is_total = bool(_KW["kw_total"].search(s))
        is_tax = bool(TAX_RE.search(s)) and not NOT_TAX_RE.search(s) and "%" not in l.text
        if DISCOUNT_RE.search(s) and not is_total:
            fin.discount += abs(v)
        elif SUBTOTAL_RE.search(s):
            fin.subtotal = v if fin.subtotal is None else fin.subtotal
        elif _KW["kw_round"].search(s) and not is_total:
            fin.rounding = v if abs(v) < 1 else fin.rounding
        elif _KW["kw_change"].search(s):
            fin.change = abs(v)
            break  # au-delà du rendu : récapitulatif TVA, pied de page -> ignoré
        elif _KW["kw_cash"].search(s) and not is_total:
            fin.cash = abs(v) if fin.cash is None else fin.cash
        elif is_tax:
            if 0 < v and (fin.total is None or v <= 0.3 * fin.total):
                taxes.append(v)
        elif is_total and not NOT_FINAL_TOTAL_RE.search(s):
            fin.labelled_totals.append(v)
        elif (first_total_kw is None and l.x1 / W > 0.6 and v > 0 and i != total_line
              and rightmost_in_row(i) and "@" not in s
              and not _KW["kw_date"].search(s) and not _KW["kw_tel"].search(s)):
            fin.items.append(v)
    fin.tax = round(sum(taxes), 2) if taxes else None
    return fin


# --------------------------------------------------------------------------- règles

RULES = {
    "missing_company": "Nom du commerçant introuvable",
    "missing_date": "Date introuvable",
    "missing_total": "Montant total introuvable",
    "date_invalid": "Date impossible (ex. 31/02 ou mois 13)",
    "date_future": "Date postérieure à la date d'analyse",
    "date_too_old": "Date antérieure à 2000",
    "totals_conflict": "Plusieurs montants « TOTAL » différents sur le reçu",
    "subtotal_tax_mismatch": "Total ≠ sous-total + taxe (+ arrondi)",
    "cash_change_mismatch": "Espèces − total ≠ rendu",
    "items_sum_mismatch": "Somme des articles ≠ total",
    "tax_rate_unusual": "Taux de taxe inhabituel (attendu 0 %, 6 % ou 10 %)",
    "total_not_max": "Le total est inférieur à un article ou au sous-total",
    "low_confidence": "Extraction peu fiable (confiance faible)",
}
# Règles « dures » : une seule suffit à signaler le document dans la baseline sans apprentissage.
HARD_RULES = {"missing_company", "missing_date", "missing_total", "date_invalid", "date_future",
              "totals_conflict", "subtotal_tax_mismatch", "cash_change_mismatch"}


@dataclass
class CheckResult:
    hits: dict            # règle -> bool
    residuals: dict       # règle -> écart numérique (0 si non applicable)
    applicable: dict      # règle -> la règle a pu être évaluée
    details: dict         # règle -> message lisible
    financials: Financials


def _min_resid(target: float, combos: list[float]) -> float:
    return min(abs(target - c) for c in combos)


def run_checks(doc: Document, ex: Extraction, today: date | None = None) -> CheckResult:
    today = today or date.today()
    fin = parse_financials(doc, ex)
    hits, res, app, det = {}, {}, {}, {}
    for r in RULES:
        hits[r], res[r], app[r], det[r] = False, 0.0, False, ""

    # champs manquants
    for f in ("company", "date", "total"):
        r = f"missing_{f}"
        app[r] = True
        hits[r] = not ex.values.get(f) or (f == "company" and len(ex.values[f]) < 3)
        if hits[r]:
            det[r] = RULES[r]

    # date
    ds = find_dates(ex.values.get("date", "")) if ex.values.get("date") else []
    if not ds:  # date non extraite mais peut-être présente ailleurs, ex. 31/02 refusée
        ds = [d for l in doc.lines for d in find_dates(l.text)][:1]
    if ds:
        s, ymd = ds[0]
        for r in ("date_invalid", "date_future", "date_too_old"):
            app[r] = True
        if ymd is None:
            hits["date_invalid"] = True
            det["date_invalid"] = f"« {s} » n'est pas une date valide"
        else:
            d = date(*ymd)
            if d > today:
                hits["date_future"] = True
                res["date_future"] = (d - today).days / 365
                det["date_future"] = f"{d.isoformat()} est dans le futur"
            if d.year < 2000:
                hits["date_too_old"] = True
                det["date_too_old"] = f"{d.isoformat()} est antérieure à 2000"

    T = fin.total
    if T is not None:
        rnd = fin.rounding or 0.0
        # plusieurs totaux étiquetés différents (hors arrondi)
        lt = [v for v in fin.labelled_totals if v > 0]
        if len(lt) >= 2:
            app["totals_conflict"] = True
            spread = max(lt) - min(lt)
            res["totals_conflict"] = spread
            if spread > max(TOL, abs(rnd) + TOL):
                hits["totals_conflict"] = True
                det["totals_conflict"] = "Totaux lus : " + ", ".join(f"{v:.2f}" for v in sorted(set(lt)))
        # sous-total + taxe
        if fin.subtotal is not None:
            app["subtotal_tax_mismatch"] = True
            tax = fin.tax or 0.0
            base = [fin.subtotal, fin.subtotal + tax, fin.subtotal + rnd, fin.subtotal + tax + rnd]
            combos = base + [b - fin.discount for b in base]
            r = _min_resid(T, combos)
            res["subtotal_tax_mismatch"] = r
            if r > TOL:
                hits["subtotal_tax_mismatch"] = True
                det["subtotal_tax_mismatch"] = (f"Total {T:.2f} vs sous-total {fin.subtotal:.2f}"
                                                + (f" + taxe {tax:.2f}" if fin.tax else ""))
            if fin.tax is not None and fin.subtotal > 0:
                app["tax_rate_unusual"] = True
                base = fin.subtotal if abs(T - fin.subtotal - fin.tax) < TOL else fin.subtotal - fin.tax
                rate = fin.tax / base if base > 0 else 0
                dev = min(abs(rate - x) for x in (0.0, 0.06, 0.10))
                res["tax_rate_unusual"] = dev
                if dev > 0.015 and fin.tax > 0.1:
                    hits["tax_rate_unusual"] = True
                    det["tax_rate_unusual"] = f"Taxe {fin.tax:.2f} = {rate:.1%} de la base {base:.2f}"
        # espèces / rendu
        if fin.cash is not None and fin.change is not None:
            app["cash_change_mismatch"] = True
            combos = [T + fin.change, T + rnd + fin.change]
            r = _min_resid(fin.cash, combos)
            res["cash_change_mismatch"] = r
            if r > TOL:
                hits["cash_change_mismatch"] = True
                det["cash_change_mismatch"] = (f"Espèces {fin.cash:.2f} − total {T:.2f} = "
                                               f"{fin.cash - T:.2f}, rendu lu {fin.change:.2f}")
        # somme des articles
        if len(fin.items) >= 1:
            app["items_sum_mismatch"] = True
            s = sum(fin.items)
            ref = [T, T - (fin.tax or 0), T - rnd, fin.subtotal or T]
            r = _min_resid(s, ref)
            res["items_sum_mismatch"] = r / max(T, 1.0)
            if r > TOL:
                hits["items_sum_mismatch"] = True
                det["items_sum_mismatch"] = f"Somme des {len(fin.items)} articles = {s:.2f} vs total {T:.2f}"
        # total plus petit qu'un article ou que le sous-total
        bigger = [v for v in fin.items + ([fin.subtotal] if fin.subtotal else []) if v > T + TOL]
        app["total_not_max"] = True
        if bigger:
            hits["total_not_max"] = True
            res["total_not_max"] = (max(bigger) - T) / max(T, 1.0)
            det["total_not_max"] = f"Montant {max(bigger):.2f} > total {T:.2f}"

    conf = [ex.confidence.get(f, 0.0) for f in ("company", "date", "total")]
    app["low_confidence"] = True
    res["low_confidence"] = 1 - min(conf)
    if min(conf) < 0.3:
        hits["low_confidence"] = True
        det["low_confidence"] = f"Confiance minimale {min(conf):.2f}"
    return CheckResult(hits, res, app, det, fin)


# --------------------------------------------------------------------------- features & modèle

def _slog(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def risk_features(doc: Document, ex: Extraction, chk: CheckResult) -> np.ndarray:
    fin = chk.financials
    v = []
    for r in RULES:
        v += [float(chk.hits[r]), float(np.log1p(abs(chk.residuals[r]))), float(chk.applicable[r])]
    # Agrégats indépendants du type de fraude : ils permettent de généraliser à une
    # manipulation jamais vue à l'entraînement (une règle « dure » inédite pèse quand même).
    arith = ("totals_conflict", "subtotal_tax_mismatch", "cash_change_mismatch")
    v += [
        float(sum(chk.hits[r] for r in HARD_RULES)),
        float(sum(chk.hits[r] for r in RULES if r not in HARD_RULES)),
        float(any(chk.hits[r] for r in ("date_invalid", "date_future", "date_too_old"))),
        float(any(chk.hits[r] for r in ("missing_company", "missing_date", "missing_total"))),
        float(sum(chk.hits[r] for r in arith)),
        float(max(np.log1p(abs(chk.residuals[r])) for r in arith)),
    ]
    T = fin.total or 0.0
    v += [
        _slog(T),
        _slog(max(fin.amounts) if fin.amounts else 0.0),
        T / max(fin.amounts) if fin.amounts and max(fin.amounts) > 0 else 0.0,
        len(fin.items),
        len(fin.amounts),
        len(doc.lines),
        float(np.mean([l.conf for l in doc.lines]) / 100) if doc.lines else 0.0,
        ex.confidence.get("company", 0.0),
        ex.confidence.get("date", 0.0),
        ex.confidence.get("total", 0.0),
        ex.confidence.get("address", 0.0),
        float(T > 0 and abs(T * 20 - round(T * 20)) > 1e-6),  # total non multiple de 0,05
    ]
    return np.nan_to_num(np.array(v, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


RARE_RULE_ALPHA = 0.002


class RiskModel:
    """Score de risque calibré (0-1) par régression logistique.

    Entrées : [règles + écarts + agrégats] + deux signaux non supervisés appris
    uniquement sur des reçus AUTHENTIQUES :
    - anomalie IsolationForest (le reçu ressemble-t-il aux reçus normaux ?) ;
    - « surprise » des règles : pour chaque règle déclenchée, p-valeur empirique de
      l'écart observé parmi les reçus authentiques, puis -log(min p) et la somme des
      -log p (méthode de Fisher). Ces agrégats sont communs à toutes les règles : une
      règle qui ne s'est jamais déclenchée sur les fraudes d'entraînement pèse quand
      même, ce qui permet de généraliser à des manipulations nouvelles.
    """

    def __init__(self, seed: int = 0):
        self.iforest = IsolationForest(n_estimators=300, contamination="auto", random_state=seed)
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")
        self.rule_names = list(RULES)

    def fit(self, X_clean: np.ndarray, X: np.ndarray, y: np.ndarray):
        # L'IsolationForest et les p-valeurs n'apprennent que ce qu'est un reçu « normal ».
        self.iforest.fit(X_clean)
        k = len(self.rule_names)
        self.clean_hits = X_clean[:, 0:3 * k:3] > 0.5
        self.clean_resid = X_clean[:, 1:3 * k:3]
        self.n_clean = len(X_clean)
        Z = self._aug(X)
        self.clf.fit(self.scaler.fit_transform(Z), y)
        return self

    def anomaly(self, X: np.ndarray) -> np.ndarray:
        return -self.iforest.score_samples(X)

    def _aug(self, X):
        logp = -np.log(self.rule_pvalues(X))
        return np.column_stack([X, self.anomaly(X), logp.max(axis=1), logp.sum(axis=1)])

    def rule_pvalues(self, X: np.ndarray) -> np.ndarray:
        """p-valeur empirique (lissée) de chaque règle déclenchée ; 1 si non déclenchée."""
        k = len(self.rule_names)
        hits = X[:, 0:3 * k:3] > 0.5
        resid = X[:, 1:3 * k:3]
        P = np.ones(hits.shape)
        for j in range(k):
            for i in np.where(hits[:, j])[0]:
                as_extreme = self.clean_hits[:, j] & (self.clean_resid[:, j] >= resid[i, j] - 1e-9)
                P[i, j] = (as_extreme.sum() + 1) / (self.n_clean + 2)
        return P

    def rare_rule(self, X: np.ndarray, alpha: float = RARE_RULE_ALPHA) -> np.ndarray:
        """Vrai si une règle s'est déclenchée avec un écart quasi jamais vu sur un reçu
        authentique (p <= alpha). Garde-fou indépendant des exemples de fraude."""
        return self.rule_pvalues(X).min(axis=1) <= alpha

    def rule_score(self, X: np.ndarray) -> np.ndarray:
        return 1 - self.rule_pvalues(X).min(axis=1)

    def learned_score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(self._aug(X)))[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.learned_score(X)


def rules_only_score(chk: CheckResult) -> float:
    """Baseline sans apprentissage : nombre de règles dures déclenchées."""
    return float(sum(chk.hits[r] for r in HARD_RULES))
