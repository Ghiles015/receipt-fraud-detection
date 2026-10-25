"""Prépare les données : split train/test reproductible + OCR Tesseract de toutes les images.

Sorties (data/processed/) :
  gt_train.jsonl / gt_test.jsonl     lignes issues des annotations SROIE (OCR parfait)
  tess_train.jsonl / tess_test.jsonl lignes issues de Tesseract sur les images

Usage : python scripts/prepare_data.py --sroie data/sroie [--workers 2]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receiptfraud.data import load_sroie, save_docs, split_ids, Document  # noqa: E402
from receiptfraud.ocr import ocr_image  # noqa: E402


def _ocr(args):
    doc_id, img_path, labels = args
    d = ocr_image(img_path, doc_id)
    d.labels = labels
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sroie", default=str(ROOT / "data" / "sroie"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-ocr", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    docs = load_sroie(Path(args.sroie))
    train_ids, test_ids = split_ids([d.doc_id for d in docs])
    split = {"train": train_ids, "test": test_ids}
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(split, indent=1))
    by_id = {d.doc_id: d for d in docs}
    for name, ids in split.items():
        save_docs([by_id[i] for i in ids], out / f"gt_{name}.jsonl")
    print(f"{len(docs)} documents : {len(train_ids)} train / {len(test_ids)} test")

    if args.skip_ocr:
        return
    t0 = time.time()
    for name, ids in split.items():
        jobs = [(i, by_id[i].image_path, by_id[i].labels) for i in ids if by_id[i].image_path]
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(_ocr, jobs, chunksize=4))
        save_docs(res, out / f"tess_{name}.jsonl")
        print(f"OCR {name}: {len(res)} images en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
