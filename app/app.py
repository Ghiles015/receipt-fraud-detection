"""Démo web : on dépose un reçu (image), on obtient les champs extraits, les contrôles
et un score de risque.

Lancement : python app/app.py  puis http://127.0.0.1:5000
"""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receiptfraud.pipeline import Analyzer  # noqa: E402

SAMPLES = ROOT / "app" / "samples"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
_analyzer = None


def analyzer() -> Analyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = Analyzer.load()
    return _analyzer


def _render(img: np.ndarray, name: str):
    report = analyzer().analyze_image(img, doc_id=name).to_dict()
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    h, w = img.shape[:2]
    return render_template("index.html", report=report, image=data_url, img_w=w, img_h=h, name=name,
                           samples=_samples(), thresholds=analyzer().thresholds)


def _samples():
    return sorted(p.name for p in SAMPLES.glob("*.jpg")) if SAMPLES.exists() else []


@app.get("/")
def index():
    return render_template("index.html", report=None, samples=_samples(), thresholds=analyzer().thresholds)


@app.post("/analyze")
def analyze():
    f = request.files.get("file")
    if not f or not f.filename:
        abort(400, "Aucun fichier reçu")
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        abort(400, "Image illisible (formats acceptés : JPG, PNG)")
    return _render(img, f.filename)


@app.get("/sample/<name>")
def sample(name):
    if name not in _samples():
        abort(404)
    return _render(cv2.imread(str(SAMPLES / name)), name)


@app.post("/api/analyze")
def api_analyze():
    """API JSON : curl -F file=@recu.jpg http://127.0.0.1:5000/api/analyze"""
    f = request.files.get("file")
    if not f:
        return {"error": "champ 'file' manquant"}, 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "image illisible"}, 400
    rep = analyzer().analyze_image(img, doc_id=f.filename).to_dict()
    rep.pop("lines")
    return rep


if __name__ == "__main__":
    app.run(debug=False, port=5000)
