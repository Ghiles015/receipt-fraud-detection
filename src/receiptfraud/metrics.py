"""Métriques d'extraction par champ.

Protocole proche de SROIE (tâche 3) : une prédiction est correcte si elle est égale
à la vérité terrain après normalisation (majuscules, espaces fusionnés).
Le total est comparé numériquement (« RM9.00 » == « 9.00 »).
- précision = correctes / prédictions non vides
- rappel    = correctes / champs présents dans la vérité terrain
On rapporte aussi la similarité de caractères moyenne (utile pour company/address,
où une virgule mal placée suffit à rendre la prédiction « fausse »).
"""
from __future__ import annotations

from collections import defaultdict

from . import FIELDS
from .textutils import find_dates, label_amount, norm, similarity


def is_correct(field: str, pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    if field == "total":
        a, b = label_amount(pred), label_amount(gold)
        return a is not None and b is not None and abs(a - b) < 0.005
    if field == "date":
        if norm(pred).replace(" ", "") == norm(gold).replace(" ", ""):
            return True
        pa, pg = find_dates(pred), find_dates(gold)
        return bool(pa and pg and pa[0][1] and pa[0][1] == pg[0][1])
    return norm(pred) == norm(gold)


def field_scores(preds: list[dict], golds: list[dict]) -> dict:
    c = defaultdict(lambda: {"tp": 0, "n_pred": 0, "n_gold": 0, "sim": 0.0, "n": 0})
    for p, g in zip(preds, golds):
        for f in FIELDS:
            s = c[f]
            pv, gv = p.get(f, ""), g.get(f, "")
            s["n_pred"] += bool(pv)
            s["n_gold"] += bool(gv)
            s["tp"] += is_correct(f, pv, gv)
            if gv:
                s["sim"] += similarity(pv, gv)
                s["n"] += 1
    out = {}
    for f in FIELDS:
        s = c[f]
        prec = s["tp"] / s["n_pred"] if s["n_pred"] else 0.0
        rec = s["tp"] / s["n_gold"] if s["n_gold"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f] = {"precision": prec, "recall": rec, "f1": f1, "char_similarity": s["sim"] / max(1, s["n"])}
    out["macro_f1"] = sum(out[f]["f1"] for f in FIELDS) / len(FIELDS)
    tp = sum(c[f]["tp"] for f in FIELDS)
    npred = sum(c[f]["n_pred"] for f in FIELDS)
    ngold = sum(c[f]["n_gold"] for f in FIELDS)
    P, R = tp / max(1, npred), tp / max(1, ngold)
    out["micro_f1"] = 2 * P * R / (P + R) if P + R else 0.0
    return out
