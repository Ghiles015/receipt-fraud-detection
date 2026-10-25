"""Génération de documents falsifiés (synthétiques) pour évaluer la détection.

Il n'existe pas de jeu public de reçus frauduleux étiquetés : on simule les
manipulations typiques d'un fraudeur (note de frais gonflée, date modifiée...)
sur des reçus authentiques. Chaque générateur renvoie None s'il ne s'applique pas.

Les falsifications sont faites au niveau texte (lignes OCR) pour l'évaluation à
grande échelle, et au niveau image (`tamper_image`) pour la démo et un test bout en bout.
"""
from __future__ import annotations

import copy
import random
import re

import cv2
import numpy as np

from .data import Document
from .features import _KW, row_context
from .labeling import line_labels
from .textutils import AMOUNT_RE, find_amounts, find_dates


def _fmt(v: float, like: str) -> str:
    s = f"{v:,.2f}" if "," in like[:-3] else f"{v:.2f}"
    return s.replace(".", ",") if like[-3] == "," else s


def _replace_amount(text: str, old_tok: str, new_val: float) -> str:
    return text.replace(old_tok, _fmt(new_val, old_tok), 1)


def _amount_lines(doc: Document):
    """(indice, token, valeur) du montant le plus à droite de chaque ligne."""
    out = []
    for i, l in enumerate(doc.lines):
        am = find_amounts(l.text)
        if am:
            out.append((i, am[-1][0], am[-1][1]))
    return out


def _total_lines(doc: Document) -> list[int]:
    lab = line_labels(doc)["total"]
    return [i for i, y in enumerate(lab) if y]


def _final_total_line(doc: Document) -> int | None:
    """La ligne du total « affiché » : celle dont la rangée contient TOTAL (hors sous-total)."""
    idx = _total_lines(doc)
    if not idx:
        return None
    ctx = row_context(doc)
    for i in idx:
        s = doc.lines[i].text + " " + ctx[i]
        if _KW["kw_total"].search(s) and not _KW["kw_subtotal"].search(s) and not _KW["kw_tax"].search(s):
            return i
    return None


def _item_lines(doc: Document) -> list[int]:
    """Lignes d'articles : montant en colonne de droite, avant le premier mot-clé TOTAL."""
    ctx = row_context(doc)
    W = max(doc.width, 1)
    first_total = next((i for i, l in enumerate(doc.lines)
                        if _KW["kw_total"].search(l.text + " " + ctx[i])), len(doc.lines))
    tot = set(_total_lines(doc))
    items = []
    for i, tok, v in _amount_lines(doc):
        s = doc.lines[i].text + " " + ctx[i]
        if i >= first_total or i in tot or v <= 0:
            continue
        if doc.lines[i].x1 / W < 0.6 or any(_KW[k].search(s) for k in ("kw_tax", "kw_cash", "kw_change", "kw_round", "kw_date")):
            continue
        items.append(i)
    return items


# --------------------------------------------------------------------------- générateurs

def total_inflated(doc: Document, rng: random.Random) -> Document | None:
    """Le total affiché est gonflé ; sous-total, taxes, espèces/rendu restent d'origine."""
    i = _final_total_line(doc)
    if i is None:
        return None
    tok, v = find_amounts(doc.lines[i].text)[-1]
    new = round(v * rng.uniform(1.15, 3.0) + rng.choice([0, 1, 5, 10]), 2)
    d = copy.deepcopy(doc)
    d.lines[i].text = _replace_amount(d.lines[i].text, tok, new)
    return d


def total_consistent(doc: Document, rng: random.Random) -> Document | None:
    """Fraude « soignée » : toutes les occurrences du total sont modifiées de façon cohérente
    (et l'espèce versée ajustée pour que le rendu reste juste). Cas difficile."""
    idx = _total_lines(doc)
    if not idx:
        return None
    _, v = find_amounts(doc.lines[idx[0]].text)[-1]
    new = round(v * rng.uniform(1.15, 2.0), 1)
    delta = new - v
    d = copy.deepcopy(doc)
    ctx = row_context(doc)
    for i, l in enumerate(d.lines):
        for tok, a in find_amounts(l.text):
            s = doc.lines[i].text + " " + ctx[i]
            if abs(a - v) < 0.005:
                l.text = _replace_amount(l.text, tok, new)
            elif _KW["kw_cash"].search(s) and not _KW["kw_change"].search(s) and a > v:
                l.text = _replace_amount(l.text, tok, a + delta)
    return d


def item_altered(doc: Document, rng: random.Random) -> Document | None:
    """Le montant d'un article est modifié sans que le total ne suive."""
    items = _item_lines(doc)
    if not items:
        return None
    i = rng.choice(items)
    tok, v = find_amounts(doc.lines[i].text)[-1]
    new = round(v * rng.uniform(2.0, 5.0) + 1, 2)
    d = copy.deepcopy(doc)
    d.lines[i].text = _replace_amount(d.lines[i].text, tok, new)
    return d


def _date_line(doc: Document) -> int | None:
    lab = line_labels(doc)["date"]
    return next((i for i, y in enumerate(lab) if y), None)


def _rewrite_date(text: str, fn) -> str | None:
    found = find_dates(text)
    if not found:
        return None
    s, ymd = found[0]
    if ymd is None:
        return None
    new = fn(s, ymd)
    return text.replace(s, new, 1) if new else None


def date_impossible(doc: Document, rng: random.Random) -> Document | None:
    """Date inexistante (31/02, mois 13...)."""
    i = _date_line(doc)
    if i is None:
        return None

    def fn(s, ymd):
        y, m, dd = ymd
        nums = re.findall(r"\d+", s)
        if len(nums) == 3 and len(nums[0]) <= 2:  # jj/mm/aaaa
            choice = rng.choice(["feb", "month"])
            if choice == "feb":
                return re.sub(r"^(\d{1,2})(\D+)(\d{1,2})", lambda m_: f"{rng.choice([30, 31])}{m_.group(2)}02", s)
            return re.sub(r"^(\d{1,2})(\D+)(\d{1,2})", lambda m_: f"{m_.group(1)}{m_.group(2)}{rng.choice([13, 14, 15])}", s)
        if len(nums) == 3 and len(nums[0]) == 4:  # aaaa-mm-jj
            return re.sub(r"^(\d{4})(\D+)(\d{1,2})(\D+)(\d{1,2})", lambda m_: f"{m_.group(1)}{m_.group(2)}02{m_.group(4)}31", s)
        return re.sub(r"^\d{1,2}", "32", s)

    t = _rewrite_date(doc.lines[i].text, fn)
    if t is None:
        return None
    d = copy.deepcopy(doc)
    d.lines[i].text = t
    return d


def date_future(doc: Document, rng: random.Random) -> Document | None:
    """Date dans le futur (année entre 2028 et 2035, donc après toute date d'analyse réaliste)."""
    i = _date_line(doc)
    if i is None:
        return None

    def fn(s, ymd):
        y = ymd[0]
        new_y = rng.randint(2028, 2035)
        if str(y) in s:
            return s.replace(str(y), str(new_y))
        yy = f"{y % 100:02d}"
        return re.sub(rf"{yy}$", f"{new_y % 100:02d}", s)

    t = _rewrite_date(doc.lines[i].text, fn)
    if t is None:
        return None
    d = copy.deepcopy(doc)
    d.lines[i].text = t
    return d


def field_removed(doc: Document, rng: random.Random) -> Document | None:
    """Suppression de l'en-tête (enseigne) ou de la date : reçu « anonymisé »."""
    lab = line_labels(doc)
    target = rng.choice(["company", "date"])
    idx = [i for i, y in enumerate(lab[target]) if y]
    if target == "date":
        idx = [i for i, l in enumerate(doc.lines) if find_dates(l.text)]
    if target == "company":
        idx = list(range(0, max(idx) + 1)) if idx else []
    if not idx:
        return None
    d = copy.deepcopy(doc)
    d.lines = [l for i, l in enumerate(d.lines) if i not in set(idx)]
    return d if d.lines else None


def tax_altered(doc: Document, rng: random.Random) -> Document | None:
    """Montant de taxe modifié (GST/SST) sans répercussion."""
    ctx = row_context(doc)
    tot = set(_total_lines(doc))
    cands = [(i, tok, v) for i, tok, v in _amount_lines(doc)
             if i not in tot and v > 0 and _KW["kw_tax"].search(doc.lines[i].text + " " + ctx[i])
             and not _KW["kw_total"].search(doc.lines[i].text + " " + ctx[i])]
    if not cands:
        return None
    i, tok, v = rng.choice(cands)
    d = copy.deepcopy(doc)
    d.lines[i].text = _replace_amount(d.lines[i].text, tok, round(v * rng.uniform(2, 4) + 0.5, 2))
    return d


GENERATORS = {
    "total_inflated": total_inflated,
    "total_consistent": total_consistent,
    "item_altered": item_altered,
    "date_impossible": date_impossible,
    "date_future": date_future,
    "field_removed": field_removed,
    "tax_altered": tax_altered,
}


def make_fraud_set(docs: list[Document], types: list[str] | None = None, seed: int = 0,
                   per_type_frac: float = 1.0) -> list[tuple[Document, str]]:
    """Applique chaque type de fraude à (une fraction de) chaque document applicable."""
    rng = random.Random(seed)
    types = types or list(GENERATORS)
    out = []
    for t in types:
        for d in docs:
            if rng.random() > per_type_frac:
                continue
            f = GENERATORS[t](d, rng)
            if f is not None:
                f.doc_id = f"{d.doc_id}__{t}"
                out.append((f, t))
    return out


# --------------------------------------------------------------------------- falsification image

def tamper_image(img: np.ndarray, box: tuple, new_text: str) -> np.ndarray:
    """Efface une zone (couleur de fond estimée) et réécrit un texte : simule une retouche."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    out = img.copy()
    pad = 2
    region = out[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    bg = np.percentile(region.reshape(-1, region.shape[-1]), 90, axis=0) if region.size else (255, 255, 255)
    cv2.rectangle(out, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), tuple(int(c) for c in bg), -1)
    h = max(8, y1 - y0)
    scale = h / 26.0
    thickness = max(1, int(round(h / 12)))
    (tw, th), _ = cv2.getTextSize(new_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x_text = max(0, x1 - tw)  # aligné à droite comme les montants
    cv2.putText(out, new_text, (x_text, y1 - max(1, (h - th) // 2)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (30, 30, 30), thickness, cv2.LINE_AA)
    return out
