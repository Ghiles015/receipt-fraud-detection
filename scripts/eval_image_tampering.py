"""Test bout en bout sur des images retouchées.

Pour chaque reçu de test, on applique une falsification texte (fraud_synth), puis on la
reporte sur l'IMAGE (zone effacée + nouveau texte dessiné). L'image passe ensuite par la
chaîne complète : Tesseract -> extraction -> contrôles -> score. C'est le scénario réel
de la démo, bruit d'OCR compris.

Sorties : reports/image_tampering_metrics.json, app/samples/*.jpg (exemples pour la démo)
"""
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receiptfraud.data import load_docs  # noqa: E402
from receiptfraud.fraud_synth import GENERATORS, tamper_image  # noqa: E402
from receiptfraud.ocr import ocr_image  # noqa: E402
from receiptfraud.pipeline import Analyzer  # noqa: E402

TYPES = ["total_inflated", "item_altered", "date_future", "date_impossible", "total_consistent"]
REF_DATE = None


def tamper(doc, ftype, seed):
    """Retourne l'image retouchée (ou None si la fraude ne s'applique pas)."""
    rng = random.Random(seed)
    fake = GENERATORS[ftype](doc, rng)
    if fake is None or len(fake.lines) != len(doc.lines):
        return None
    img = cv2.imread(doc.image_path)
    changed = 0
    for a, b in zip(doc.lines, fake.lines):
        if a.text != b.text:
            img = tamper_image(img, (a.x0, a.y0, a.x1, a.y1), b.text)
            changed += 1
    return img if changed else None


def _ocr_job(args):
    doc_id, img_bytes = args
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    return ocr_image(img, doc_id)


def main():
    import joblib
    bundle = joblib.load(ROOT / "models" / "risk_model.joblib")
    an = Analyzer.load(today=bundle["ref_date"])
    test_gt = load_docs(ROOT / "data" / "processed" / "gt_test.jsonl")
    test_tess = {d.doc_id: d for d in load_docs(ROOT / "data" / "processed" / "tess_test.jsonl")}

    jobs, meta = [], []
    for i, d in enumerate(test_gt):
        for t in TYPES:
            img = tamper(d, t, seed=i)
            if img is None:
                continue
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            jobs.append((f"{d.doc_id}__{t}", buf.tobytes()))
            meta.append((d.doc_id, t))
    with ProcessPoolExecutor(2) as ex:
        docs = list(ex.map(_ocr_job, jobs, chunksize=4))

    scores = {"clean": [an.analyze_document(test_tess[d.doc_id]).risk_score for d in test_gt if d.doc_id in test_tess]}
    for (doc_id, t), d in zip(meta, docs):
        scores.setdefault(t, []).append(an.analyze_document(d).risk_score)

    th = an.thresholds
    res = {"thresholds": th}
    for t, s in scores.items():
        s = np.array(s)
        res[t] = {"n": len(s), "flag_rate@medium": float((s >= th["medium"]).mean()),
                  "flag_rate@high": float((s >= th["high"]).mean()), "median_score": float(np.median(s))}
    from sklearn.metrics import roc_auc_score
    c = np.array(scores["clean"])
    for t in TYPES:
        if t in scores:
            f = np.array(scores[t])
            res[t]["roc_auc_vs_clean"] = float(roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(f))], np.r_[c, f]))
    (ROOT / "reports" / "image_tampering_metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"{'type':18s}{'n':>5s}{'signalé@moyen':>15s}{'signalé@élevé':>15s}{'AUC':>7s}")
    for t, r in res.items():
        if t == "thresholds":
            continue
        print(f"{t:18s}{r['n']:5d}{r['flag_rate@medium']:15.1%}{r['flag_rate@high']:15.1%}"
              f"{r.get('roc_auc_vs_clean', float('nan')):7.3f}")

    # exemples pour la démo : 2 authentiques + 3 retouchés
    out = ROOT / "app" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.jpg"):
        p.unlink()
    picked = {"total_inflated": 1, "date_future": 1, "item_altered": 1}
    for i, d in enumerate(test_gt):
        if sum(picked.values()) == 0:
            break
        for t, k in picked.items():
            if k and (img := tamper(d, t, seed=i)) is not None:
                rep = an.analyze_image(img)
                if rep.risk_level != "faible":
                    cv2.imwrite(str(out / f"{d.doc_id}_falsifie_{t}.jpg"), img)
                    picked[t] = 0
                    break
    n = 0
    for d in test_gt:
        rep = an.analyze_document(test_tess[d.doc_id])
        if rep.risk_level == "faible" and all(rep.fields.values()) and n < 2:
            cv2.imwrite(str(out / f"{d.doc_id}_authentique.jpg"), cv2.imread(d.image_path))
            n += 1
    print("exemples :", sorted(p.name for p in out.glob("*.jpg")))


if __name__ == "__main__":
    main()
