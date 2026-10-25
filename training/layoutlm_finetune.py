"""Fine-tuning de LayoutLM (Hugging Face) comme extracteur alternatif.

À lancer sur GPU (Google Colab suffit, ~10 min pour 10 époques) :
    pip install torch transformers
    python training/layoutlm_finetune.py --epochs 10

Le modèle prédit, pour chaque token, le champ de sa ligne (O / company / date / address / total).
Les probabilités sont moyennées par ligne puis passées au MÊME décodeur et aux MÊMES métriques
que l'extracteur classique (src/receiptfraud/extract.py) : la comparaison est donc directe.

Non exécuté dans l'environnement de développement (pas de GPU ni d'accès au hub Hugging Face) :
les résultats LayoutLM ne figurent donc pas dans le README tant que vous ne l'avez pas lancé.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receiptfraud import FIELDS  # noqa: E402
from receiptfraud.data import Document, load_docs  # noqa: E402
from receiptfraud.extract import decode  # noqa: E402
from receiptfraud.labeling import line_labels  # noqa: E402
from receiptfraud.metrics import field_scores  # noqa: E402

LABELS = ["O"] + list(FIELDS)
PRIORITY = ["total", "date", "company", "address"]  # si une ligne porte plusieurs champs
MODEL_NAME = "microsoft/layoutlm-base-uncased"


def doc_words(doc: Document):
    """Découpe chaque ligne en mots, avec une boîte par mot (interpolation horizontale)."""
    words, boxes, line_idx = [], [], []
    W, H = max(doc.width, 1), max(doc.height, 1)
    for i, l in enumerate(doc.lines):
        ws = l.text.split()
        if not ws:
            continue
        total_chars = sum(len(w) for w in ws) + len(ws) - 1
        x = l.x0
        for w in ws:
            wx1 = x + (l.x1 - l.x0) * len(w) / max(1, total_chars)
            box = [int(1000 * x / W), int(1000 * l.y0 / H), int(1000 * wx1 / W), int(1000 * l.y1 / H)]
            boxes.append([min(1000, max(0, v)) for v in box])
            words.append(w)
            line_idx.append(i)
            x = wx1 + (l.x1 - l.x0) / max(1, total_chars)
    return words, boxes, line_idx


class ReceiptDataset(Dataset):
    def __init__(self, docs, tokenizer, max_len=512):
        self.items = []
        for d in docs:
            words, boxes, line_idx = doc_words(d)
            lab = line_labels(d) if d.labels else None
            enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            word_ids = enc.word_ids(0)
            bbox, labels, tok_line = [], [], []
            for wid in word_ids:
                if wid is None:
                    bbox.append([0, 0, 0, 0])
                    labels.append(-100)
                    tok_line.append(-1)
                    continue
                bbox.append(boxes[wid])
                li = line_idx[wid]
                tok_line.append(li)
                y = 0
                if lab:
                    for f in PRIORITY:
                        if lab[f][li]:
                            y = LABELS.index(f)
                            break
                labels.append(y)
            self.items.append({
                "input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0],
                "token_type_ids": enc["token_type_ids"][0], "bbox": torch.tensor(bbox),
                "labels": torch.tensor(labels), "tok_line": torch.tensor(tok_line),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class LayoutLMExtractor:
    def __init__(self, model, tokenizer, device):
        self.model, self.tok, self.device = model.eval(), tokenizer, device

    @torch.no_grad()
    def predict_probs(self, doc):
        item = ReceiptDataset([doc], self.tok).items[0]
        batch = {k: v[None].to(self.device) for k, v in item.items() if k not in ("labels", "tok_line")}
        probs = torch.softmax(self.model(**batch).logits[0], -1).cpu().numpy()
        n = len(doc.lines)
        out = {f: np.zeros(n) for f in FIELDS}
        cnt = np.zeros(n)
        for t, li in enumerate(item["tok_line"].tolist()):
            if li < 0:
                continue
            cnt[li] += 1
            for j, f in enumerate(FIELDS):
                out[f][li] += probs[t, j + 1]
        for f in FIELDS:
            out[f] = out[f] / np.maximum(cnt, 1)
        return out

    def extract(self, doc):
        return decode(doc, self.predict_probs(doc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data/processed/gt_train.jsonl"))
    ap.add_argument("--test", nargs="+", default=[str(ROOT / "data/processed/gt_test.jsonl"),
                                                  str(ROOT / "data/processed/tess_test.jsonl")])
    ap.add_argument("--extra-train", default=str(ROOT / "data/processed/tess_train.jsonl"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "models/layoutlm"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = LayoutLMTokenizerFast.from_pretrained(MODEL_NAME)
    model = LayoutLMForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS)).to(device)

    train_docs = load_docs(Path(args.train))
    if args.extra_train and Path(args.extra_train).exists():
        train_docs += load_docs(Path(args.extra_train))
    ds = ReceiptDataset(train_docs, tok)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)
    # pondération : la classe O est très majoritaire
    counts = np.bincount(torch.cat([x["labels"] for x in ds.items]).clamp(min=0).numpy(), minlength=len(LABELS))
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights / weights.mean(), ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(dl) * args.epochs), len(dl) * args.epochs)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for b in dl:
            b = {k: v.to(device) for k, v in b.items() if k != "tok_line"}
            labels = b.pop("labels")
            logits = model(**b).logits
            loss = loss_fn(logits.view(-1, len(LABELS)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += loss.item()
        print(f"époque {ep + 1}/{args.epochs} - perte {tot / len(dl):.4f}")

    ext = LayoutLMExtractor(model, tok, device)
    results = {}
    for path in args.test:
        docs = load_docs(Path(path))
        preds = [ext.extract(d).values for d in docs]
        results[Path(path).stem] = s = field_scores(preds, [d.labels for d in docs])
        print(Path(path).stem, {f: round(s[f]["f1"], 3) for f in FIELDS}, "macro", round(s["macro_f1"], 3))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (ROOT / "reports" / "layoutlm_metrics.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
