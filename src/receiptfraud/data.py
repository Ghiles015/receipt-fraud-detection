"""Chargement du jeu SROIE et représentation commune d'un document OCR.

Un document est une liste de `Line` (texte + boîte englobante), quelle que soit
la source : annotations SROIE (OCR « parfait ») ou sortie de Tesseract.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import FIELDS


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 100.0  # confiance OCR (100 pour la vérité terrain)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return max(1.0, self.y1 - self.y0)


@dataclass
class Document:
    doc_id: str
    lines: list[Line]
    width: float
    height: float
    labels: dict = field(default_factory=dict)  # vérité terrain des champs
    image_path: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Document":
        return Document(
            doc_id=d["doc_id"],
            lines=[Line(**l) for l in d["lines"]],
            width=d["width"],
            height=d["height"],
            labels=d.get("labels", {}),
            image_path=d.get("image_path"),
        )


def sort_reading_order(lines: list[Line]) -> list[Line]:
    """Trie les lignes de haut en bas puis de gauche à droite (tolérance d'une demi-hauteur)."""
    lines = sorted(lines, key=lambda l: (l.cy, l.x0))
    out: list[Line] = []
    row: list[Line] = []
    for l in lines:
        if row and abs(l.cy - row[0].cy) > 0.5 * min(l.h, row[0].h):
            out.extend(sorted(row, key=lambda r: r.x0))
            row = []
        row.append(l)
    out.extend(sorted(row, key=lambda r: r.x0))
    return out


def parse_box_file(path: Path) -> list[Line]:
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(",", 8)
        if len(parts) < 9:
            continue
        try:
            xs = [float(parts[i]) for i in (0, 2, 4, 6)]
            ys = [float(parts[i]) for i in (1, 3, 5, 7)]
        except ValueError:
            continue
        text = parts[8].strip()
        if not text:
            continue
        lines.append(Line(text, min(xs), min(ys), max(xs), max(ys)))
    return sort_reading_order(lines)


def load_sroie(root: Path) -> list[Document]:
    """Charge les documents SROIE (dossiers box/, key/, img/)."""
    root = Path(root)
    docs = []
    for key_path in sorted((root / "key").glob("*.json")):
        doc_id = key_path.stem
        box_path = root / "box" / f"{doc_id}.csv"
        if not box_path.exists():
            continue
        labels = json.loads(key_path.read_text(encoding="utf-8", errors="ignore"))
        labels = {k: (labels.get(k) or "").strip() for k in FIELDS}
        lines = parse_box_file(box_path)
        if not lines:
            continue
        width = max(l.x1 for l in lines) + min(l.x0 for l in lines)
        height = max(l.y1 for l in lines) + min(l.y0 for l in lines)
        img = root / "img" / f"{doc_id}.jpg"
        docs.append(Document(doc_id, lines, width, height, labels, str(img) if img.exists() else None))
    return docs


def split_ids(ids: list[str], test_frac: float = 0.2, seed: int = 42) -> tuple[list[str], list[str]]:
    ids = sorted(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_frac))
    return sorted(ids[n_test:]), sorted(ids[:n_test])


def save_docs(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")


def load_docs(path: Path) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        return [Document.from_dict(json.loads(l)) for l in f if l.strip()]
