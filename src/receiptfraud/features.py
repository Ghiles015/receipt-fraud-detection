"""Features de mise en page et de texte par ligne OCR."""
from __future__ import annotations

import re

import numpy as np

from .data import Document, Line
from .textutils import find_amounts, find_dates

KEYWORDS = {
    "kw_sdn": r"\b(SDN|BHD|S/B|ENTERPRISE|TRADING|RESTORAN|RESTAURANT|STORE|MART|SHOP|CAFE|HARDWARE|BOOK|MARKET|STATIONERY|KEDAI)\b",
    "kw_addr": r"\b(JALAN|JLN|TAMAN|TMN|NO\.?|LOT|LORONG|BANDAR|KUALA|LUMPUR|SELANGOR|JOHOR|KLANG|PETALING|JAYA|MALAYSIA|\d{5})\b",
    "kw_tel": r"\b(TEL|FAX|PHONE|H/P|EMAIL|WWW|GST\s*(ID|REG|NO)|REG\.?\s*NO|CO\.?\s*NO)\b",
    "kw_total": r"\bTOTAL\b",
    "kw_subtotal": r"\b(SUB\s*-?\s*TOTAL|SUBTOTAL|NET\s*TOTAL|TOTAL\s*EXCL)\b",
    "kw_grand": r"\b(GRAND|ROUNDED|ROUND\w*\s*TOTAL|NETT|AMOUNT\s*DUE|TOTAL\s*\(?RM|TOTAL\s*AMOUNT|TOTAL\s*INCL)\b",
    "kw_round": r"\bROUND",
    "kw_tax": r"\b(GST|SST|TAX|SR|ZR|SERVICE\s*CHARGE)\b",
    "kw_cash": r"\b(CASH|TENDER|PAID|PAYMENT|VISA|MASTER|CARD|CREDIT|DEBIT)\b",
    "kw_change": r"\b(CHANGE|BALANCE|BAL)\b",
    "kw_date": r"\b(DATE|TARIKH)\b",
    "kw_time": r"\d{1,2}:\d{2}",
    "kw_qty": r"\b(QTY|PCS?|UNIT|X\s*\d)\b",
    "kw_rm": r"\bRM\b",
    "kw_invoice": r"\b(INVOICE|RECEIPT|BILL|DOC|TAX\s*INVOICE)\b",
    "kw_thanks": r"\b(THANK|PLEASE|WELCOME|GOODS|RETURN|EXCHANGE)\b",
}
_KW = {k: re.compile(v, re.I) for k, v in KEYWORDS.items()}
CONTEXT_KW = ["kw_total", "kw_subtotal", "kw_grand", "kw_round", "kw_tax", "kw_cash", "kw_change", "kw_date", "kw_qty"]


def _same_row(a: Line, b: Line) -> bool:
    return abs(a.cy - b.cy) < 0.6 * max(a.h, b.h)


def row_context(doc: Document) -> list[str]:
    """Texte des autres segments de la même rangée (à gauche), pour relier « TOTAL : » à « 9.00 »."""
    ctx = []
    for i, l in enumerate(doc.lines):
        left = [o for j, o in enumerate(doc.lines) if j != i and _same_row(l, o) and o.x1 <= l.x0 + 5]
        ctx.append(" ".join(o.text for o in sorted(left, key=lambda o: o.x0)))
    return ctx


FEATURE_NAMES: list[str] = []


def line_features(doc: Document) -> np.ndarray:
    lines = doc.lines
    n = len(lines)
    W, H = max(doc.width, 1.0), max(doc.height, 1.0)
    heights = np.array([l.h for l in lines]) if lines else np.array([1.0])
    med_h = float(np.median(heights))
    ctx = row_context(doc)
    all_amounts = [v for l in lines for _, v in find_amounts(l.text)]
    max_amt = max(all_amounts) if all_amounts else 0.0
    sorted_amts = sorted(set(all_amounts), reverse=True)
    rows = []
    names = []
    for i, l in enumerate(lines):
        t = l.text
        f = {}
        f["x0"], f["x1"] = l.x0 / W, l.x1 / W
        f["y0"], f["y1"] = l.y0 / H, l.y1 / H
        f["w"] = (l.x1 - l.x0) / W
        f["h_rel"] = l.h / med_h
        f["idx"] = i / max(1, n - 1)
        f["idx_abs"] = min(i, 30)
        f["n_lines"] = n
        f["len"] = len(t)
        f["n_tok"] = len(t.split())
        nc = max(1, len(t))
        f["frac_digit"] = sum(c.isdigit() for c in t) / nc
        f["frac_alpha"] = sum(c.isalpha() for c in t) / nc
        f["frac_upper"] = sum(c.isupper() for c in t) / max(1, sum(c.isalpha() for c in t))
        f["frac_punct"] = sum(not c.isalnum() and not c.isspace() for c in t) / nc
        f["conf"] = l.conf / 100
        am = find_amounts(t)
        f["n_amounts"] = len(am)
        v = max((a for _, a in am), default=0.0)
        f["amount"] = float(np.sign(v) * np.log1p(abs(v)))
        f["is_max_amount"] = float(bool(am) and abs(v - max_amt) < 1e-6)
        f["amount_rank"] = (sorted_amts.index(v) if am and v in sorted_amts else 99)
        f["n_amount_occ"] = sum(abs(a - v) < 1e-6 for a in all_amounts) if am else 0
        f["ends_with_amount"] = float(bool(re.search(r"\d[.,]\d{2}\s*$", t)))
        d = find_dates(t)
        f["has_date"] = float(bool(d))
        f["date_valid"] = float(any(x[1] for x in d))
        for k, rx in _KW.items():
            f[k] = float(bool(rx.search(t)))
        c = ctx[i]
        for k in CONTEXT_KW:
            f["ctx_" + k] = float(bool(_KW[k].search(c)))
        prev_t = lines[i - 1].text if i > 0 else ""
        f["prev_kw_total"] = float(bool(_KW["kw_total"].search(prev_t)))
        f["prev_kw_date"] = float(bool(_KW["kw_date"].search(prev_t)))
        f["above_first_date"] = 0.0
        rows.append(f)
    # position relative à la première date (l'en-tête est au-dessus)
    first_date = next((i for i, r in enumerate(rows) if r["has_date"]), n)
    for i, r in enumerate(rows):
        r["above_first_date"] = float(i < first_date)
    if rows:
        names = list(rows[0].keys())
        if not FEATURE_NAMES:
            FEATURE_NAMES.extend(names)
        return np.array([[r[k] for k in names] for r in rows], dtype=np.float32)
    return np.zeros((0, len(FEATURE_NAMES) or 1), dtype=np.float32)
