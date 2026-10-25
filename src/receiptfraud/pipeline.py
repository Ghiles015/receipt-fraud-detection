"""Pipeline complet : image (ou document OCR) -> champs extraits -> contrôles -> score de risque."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np

from . import FIELDS
from .data import Document
from .fraud import RULES, RiskModel, risk_features, run_checks
from .ocr import ocr_image

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass
class Report:
    doc_id: str
    fields: dict
    confidence: dict
    checks: list           # [{rule, label, hit, detail}]
    risk_score: float      # 0-100
    risk_level: str        # faible / moyen / élevé
    anomaly: float
    financials: dict
    lines: list            # [{text, box, field}] pour l'affichage

    def to_dict(self):
        return self.__dict__


def risk_level(score: float, thresholds: dict) -> str:
    if score >= thresholds["high"]:
        return "élevé"
    if score >= thresholds["medium"]:
        return "moyen"
    return "faible"


class Analyzer:
    def __init__(self, extractor, risk_model: RiskModel | None = None, thresholds: dict | None = None,
                 today: date | None = None):
        self.extractor = extractor
        self.risk_model = risk_model
        self.thresholds = thresholds or {"medium": 30.0, "high": 60.0}
        self.today = today

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR, today: date | None = None) -> "Analyzer":
        ext = joblib.load(model_dir / "extractor.joblib")
        bundle = joblib.load(model_dir / "risk_model.joblib") if (model_dir / "risk_model.joblib").exists() else {}
        return cls(ext, bundle.get("model"), bundle.get("thresholds"), today)

    def analyze_document(self, doc: Document) -> Report:
        ex = self.extractor.extract(doc)
        chk = run_checks(doc, ex, self.today)
        x = risk_features(doc, ex, chk)[None, :]
        rare = False
        if self.risk_model is not None:
            score = float(self.risk_model.predict_proba(x)[0] * 100)
            anomaly = float(self.risk_model.anomaly(x)[0])
            rare = bool(self.risk_model.rare_rule(x)[0])
        else:
            score = min(100.0, 35.0 * sum(chk.hits.values()))
            anomaly = float("nan")
        field_of = {}
        for f in FIELDS:
            for i in ex.lines.get(f, []):
                field_of[i] = f
        lines = [{"text": l.text, "box": [l.x0, l.y0, l.x1, l.y1], "field": field_of.get(i)}
                 for i, l in enumerate(doc.lines)]
        checks = [{"rule": r, "label": RULES[r], "hit": bool(chk.hits[r]), "applicable": bool(chk.applicable[r]),
                   "detail": chk.details[r]} for r in RULES]
        fin = {k: v for k, v in chk.financials.__dict__.items() if k != "amounts"}
        level = risk_level(score, self.thresholds)
        if rare and level == "faible":
            level = "moyen"  # une règle rarissime sur des reçus authentiques suffit à alerter
        return Report(doc.doc_id, dict(ex.values), {k: round(v, 3) for k, v in ex.confidence.items()},
                      checks, round(score, 1), level,
                      None if np.isnan(anomaly) else round(anomaly, 3), fin, lines)

    def analyze_image(self, img_or_path, doc_id: str = "upload") -> Report:
        return self.analyze_document(ocr_image(img_or_path, doc_id))
