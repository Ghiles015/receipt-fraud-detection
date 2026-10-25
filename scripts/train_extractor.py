"""Entraîne l'extracteur et l'évalue (F1 par champ) contre la baseline à règles.

Configurations évaluées sur le jeu de test (125 reçus jamais vus) :
  - entrée « gt »   : boîtes et texte des annotations SROIE (OCR parfait)
  - entrée « tess » : sortie de Tesseract sur les images (conditions réelles)
Modèles : rules (baseline), ml_gt (entraîné sur OCR parfait), ml_mix (entraîné sur gt + Tesseract).

Sorties : models/extractor.joblib, reports/extraction_metrics.json, reports/extraction_errors.csv
"""
import csv
import json
import sys
import time
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receiptfraud import FIELDS  # noqa: E402
from receiptfraud.data import load_docs  # noqa: E402
from receiptfraud.extract import LineClassifierExtractor, RuleExtractor  # noqa: E402
from receiptfraud.labeling import line_labels  # noqa: E402
from receiptfraud.metrics import field_scores, is_correct  # noqa: E402
from receiptfraud.textutils import clean_company, norm, similarity  # noqa: E402

P = ROOT / "data" / "processed"


def oracle(docs):
    """Plafond atteignable par une sélection parfaite de lignes (bruit OCR/annotation)."""
    preds = []
    for d in docs:
        lab = line_labels(d)
        p = {}
        for f in FIELDS:
            sel = [l.text for l, y in zip(d.lines, lab[f]) if y]
            p[f] = (clean_company(" ".join(sel)) if f == "company" else norm(" ".join(sel))) if sel else ""
        p["date"] = d.labels["date"] if any(lab["date"]) else ""
        p["total"] = d.labels["total"] if any(lab["total"]) else ""
        preds.append(p)
    return field_scores(preds, [d.labels for d in docs])


def error_rows(model_name, input_name, docs, exs):
    rows = []
    lab_cache = {}
    for d, ex in zip(docs, exs):
        for f in FIELDS:
            pv, gv = ex.values.get(f, ""), d.labels.get(f, "")
            if not gv or is_correct(f, pv, gv):
                continue
            if d.doc_id not in lab_cache:
                lab_cache[d.doc_id] = line_labels(d)
            gold_lines = {i for i, y in enumerate(lab_cache[d.doc_id][f]) if y}
            used = set(ex.lines.get(f, []))
            if not pv:
                cat = "non_extrait"
            elif not gold_lines:
                cat = "champ_absent_de_l_ocr"
            elif used & gold_lines and similarity(pv, gv) >= 0.85:
                cat = "bonne_ligne_texte_imparfait"
            elif used & gold_lines:
                cat = "bonne_ligne_partiel"
            else:
                cat = "mauvaise_ligne"
            rows.append({"model": model_name, "input": input_name, "doc_id": d.doc_id, "field": f,
                         "category": cat, "pred": pv, "gold": gv, "similarity": round(similarity(pv, gv), 3)})
    return rows


def main():
    sets = {k: load_docs(P / f"{k}.jsonl") for k in ("gt_train", "gt_test", "tess_train", "tess_test")
            if (P / f"{k}.jsonl").exists()}
    results, errors = {}, []
    t0 = time.time()
    models = {"rules": RuleExtractor()}
    models["ml_gt"] = LineClassifierExtractor().fit(sets["gt_train"])
    print(f"ml_gt entraîné en {time.time() - t0:.0f}s")
    if "tess_train" in sets:
        t0 = time.time()
        models["ml_mix"] = LineClassifierExtractor().fit(sets["gt_train"] + sets["tess_train"])
        print(f"ml_mix entraîné en {time.time() - t0:.0f}s")

    for inp in ("gt", "tess"):
        test = sets.get(f"{inp}_test")
        if not test:
            continue
        results[f"oracle@{inp}"] = oracle(test)
        for name, m in models.items():
            exs = [m.extract(d) for d in test]
            results[f"{name}@{inp}"] = field_scores([e.values for e in exs], [d.labels for d in test])
            errors += error_rows(name, inp, test, exs)

    print(f"\n{'config':18s}" + "".join(f"{f:>10s}" for f in FIELDS) + f"{'macro':>8s}")
    for k, s in results.items():
        print(f"{k:18s}" + "".join(f"{s[f]['f1']:10.3f}" for f in FIELDS) + f"{s['macro_f1']:8.3f}")

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "extraction_metrics.json").write_text(json.dumps(results, indent=1))
    with open(ROOT / "reports" / "extraction_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(errors[0].keys()))
        w.writeheader()
        w.writerows(errors)
    best = "ml_mix" if "ml_mix" in models else "ml_gt"
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(models[best], ROOT / "models" / "extractor.joblib", compress=3)
    print(f"\nModèle sauvegardé : models/extractor.joblib ({best})")


if __name__ == "__main__":
    main()
