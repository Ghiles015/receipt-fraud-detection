"""Projection des étiquettes de champs (niveau document) sur les lignes OCR.

SROIE ne fournit que la valeur de chaque champ par reçu. Pour entraîner un
classifieur de lignes, on étiquette chaque ligne qui « porte » un champ :
- company / address : forte couverture des tokens de la ligne par la valeur du champ
- date : la ligne contient la chaîne de date (espaces ignorés)
- total : la ligne contient un montant égal au total
"""
from __future__ import annotations

from difflib import SequenceMatcher

from .data import Document
from .textutils import find_amounts, label_amount, norm, norm_nospace, find_dates


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def _greedy_span(lines: list[str], target: str, exclude: set[int], max_lines: int = 8,
                 min_ratio: float = 0.7) -> set[int]:
    """Trouve un bloc contigu de lignes dont la concaténation ressemble le plus au champ.
    Robuste aux petites différences OCR entre les boîtes et la vérité terrain."""
    if not target:
        return set()
    cand = [i for i in range(len(lines)) if i not in exclude and len(norm_nospace(lines[i])) >= 2]
    if not cand:
        return set()
    # point de départ : ligne la plus « contenue » dans le champ, pondérée par sa longueur
    def seed_score(i):
        t = norm(lines[i])
        m = SequenceMatcher(None, t, norm(target))
        matched = sum(b.size for b in m.get_matching_blocks())
        in_field = matched / max(1, len(t))          # part de la ligne présente dans le champ
        field_in = matched / max(1, len(norm(target)))  # part du champ présente dans la ligne
        return field_in * (in_field >= 0.8 or field_in >= 0.9)
    start = max(cand, key=seed_score)
    if seed_score(start) == 0:
        return set()
    span = [start]
    best = _ratio(lines[start], target)
    improved = True
    while improved and len(span) < max_lines:
        improved = False
        for j in (min(span) - 1, max(span) + 1):
            if j < 0 or j >= len(lines) or j in exclude:
                continue
            new = sorted(span + [j])
            r = _ratio(" ".join(lines[k] for k in new), target)
            if r > best + 1e-6:
                best, span, improved = r, new, True
                break
    return set(span) if best >= min_ratio else set()


def line_labels(doc: Document) -> dict[str, list[int]]:
    lab = doc.labels
    n = len(doc.lines)
    out = {f: [0] * n for f in ("company", "date", "address", "total")}
    total_val = label_amount(lab.get("total", "")) if lab.get("total") else None
    date_ns = norm_nospace(lab.get("date", ""))
    gt_date = find_dates(lab.get("date", ""))
    gt_ymd = gt_date[0][1] if gt_date else None
    texts = [l.text for l in doc.lines]
    comp = _greedy_span(texts, lab.get("company", ""), set())
    addr = _greedy_span(texts, lab.get("address", ""), comp)
    for i in comp:
        out["company"][i] = 1
    for i in addr:
        out["address"][i] = 1
    for i, l in enumerate(doc.lines):
        t = l.text
        nt = norm_nospace(t)
        # date : chaîne exacte, ou date de même valeur (OCR Tesseract)
        if date_ns and date_ns in nt:
            out["date"][i] = 1
        elif gt_ymd and any(ymd == gt_ymd for _, ymd in find_dates(t)):
            out["date"][i] = 1
        # total
        if total_val is not None and any(abs(v - total_val) < 0.005 for _, v in find_amounts(t)):
            out["total"][i] = 1
    return out
