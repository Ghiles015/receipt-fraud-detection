"""OCR avec Tesseract : image -> lignes de texte avec boîtes englobantes.

PaddleOCR est supporté en option (si installé) via `engine="paddle"`.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .data import Document, Line, sort_reading_order


def preprocess(img: np.ndarray) -> np.ndarray:
    """Niveaux de gris + agrandissement des petites images (Tesseract préfère ~300 dpi)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    if w < 1000:
        scale = 1000 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def _split_on_gaps(words: list[dict], gap_factor: float = 2.5) -> list[list[dict]]:
    """Coupe une ligne Tesseract en segments quand l'écart horizontal est grand
    (ex. « TOTAL ............ 9.00 » -> deux segments, comme les annotations SROIE)."""
    words = sorted(words, key=lambda w: w["x0"])
    if not words:
        return []
    heights = [w["y1"] - w["y0"] for w in words]
    med_h = float(np.median(heights)) if heights else 10.0
    segs = [[words[0]]]
    for w in words[1:]:
        if w["x0"] - segs[-1][-1]["x1"] > gap_factor * med_h:
            segs.append([w])
        else:
            segs[-1].append(w)
    return segs


def tesseract_lines(img: np.ndarray, psm: int = 6) -> tuple[list[Line], float, float]:
    import pytesseract

    proc = preprocess(img)
    scale = proc.shape[1] / img.shape[1]
    data = pytesseract.image_to_data(proc, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        conf = float(data["conf"][i])
        if not txt or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups[key].append({
            "text": txt,
            "conf": conf,
            "x0": data["left"][i] / scale,
            "y0": data["top"][i] / scale,
            "x1": (data["left"][i] + data["width"][i]) / scale,
            "y1": (data["top"][i] + data["height"][i]) / scale,
        })
    lines = []
    for words in groups.values():
        for seg in _split_on_gaps(words):
            lines.append(Line(
                text=" ".join(w["text"] for w in seg),
                x0=min(w["x0"] for w in seg),
                y0=min(w["y0"] for w in seg),
                x1=max(w["x1"] for w in seg),
                y1=max(w["y1"] for w in seg),
                conf=float(np.mean([w["conf"] for w in seg])),
            ))
    return sort_reading_order(lines), float(img.shape[1]), float(img.shape[0])


def paddle_lines(img: np.ndarray) -> tuple[list[Line], float, float]:  # pragma: no cover - optionnel
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    res = ocr.ocr(img, cls=True)[0] or []
    lines = []
    for box, (txt, conf) in res:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lines.append(Line(txt, min(xs), min(ys), max(xs), max(ys), conf * 100))
    return sort_reading_order(lines), float(img.shape[1]), float(img.shape[0])


def ocr_image(img_or_path, doc_id: str = "doc", engine: str = "tesseract") -> Document:
    if isinstance(img_or_path, (str, Path)):
        img = cv2.imread(str(img_or_path))
        if img is None:
            raise ValueError(f"Image illisible : {img_or_path}")
        path = str(img_or_path)
    else:
        img, path = img_or_path, None
    fn = paddle_lines if engine == "paddle" else tesseract_lines
    lines, w, h = fn(img)
    return Document(doc_id, lines, w, h, image_path=path)
