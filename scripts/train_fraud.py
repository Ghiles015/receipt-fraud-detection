"""Entraîne et évalue le module de détection de fraude.

Protocole :
- Les fraudes sont synthétiques (7 types, cf. fraud_synth.py), générées sur les reçus
  authentiques. Entraînement : reçus de train ; évaluation : reçus de test (jamais vus,
  graine différente).
- Pour que le modèle de risque voie des extractions réalistes, les reçus de train sont
  extraits « hors pli » (extracteur entraîné sans le reçu concerné, 5 plis).
- Les seuils de décision sont fixés sur les reçus authentiques de train
  (taux de faux positifs visé : 10 % pour « moyen », 3 % pour « élevé »), puis appliqués au test.
- Généralisation : modèle « leave-one-type-out » (le type évalué est absent de l'entraînement).

Sorties : models/risk_model.joblib, reports/fraud_metrics.json, reports/figures/*.png
"""
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receiptfraud.data import load_docs  # noqa: E402
from receiptfraud.extract import LineClassifierExtractor  # noqa: E402
from receiptfraud.fraud import HARD_RULES, RULES, RiskModel, risk_features, rules_only_score, run_checks  # noqa: E402
from receiptfraud.fraud_synth import GENERATORS, make_fraud_set  # noqa: E402

P = ROOT / "data" / "processed"
REF_DATE = date(2026, 10, 25)  # date d'analyse fixée pour la reproductibilité
FPR_MEDIUM, FPR_HIGH = 0.10, 0.03


def extract_all(extractor, items):
    return [(d, t, extractor.extract(d)) for d, t in items]


def featurize(extracted):
    """extracted : [(doc, type, extraction)] avec type = 'clean' ou un type de fraude."""
    X, R, types, hits = [], [], [], []
    for d, t, ex in extracted:
        chk = run_checks(d, ex, REF_DATE)
        X.append(risk_features(d, ex, chk))
        R.append(rules_only_score(chk))
        types.append(t)
        hits.append({r: bool(chk.hits[r]) for r in RULES})
    return np.array(X), np.array(R), np.array(types), hits


def build_items(docs, seed, frac):
    return [(d, "clean") for d in docs] + make_fraud_set(docs, seed=seed, per_type_frac=frac)


def oof_train_extractions(gt_train, tess_train, k=5):
    cache = P / "oof_extractions.joblib"
    if cache.exists():
        return joblib.load(cache)
    ids = sorted({d.doc_id for d in gt_train})
    rng = np.random.RandomState(0)
    fold = {i: f for i, f in zip(ids, rng.permutation(len(ids)) % k)}
    parts = []
    for f in range(k):
        t0 = time.time()
        tr = [d for d in gt_train + tess_train if fold[d.doc_id] != f]
        ext = LineClassifierExtractor(n_folds=3).fit(tr)
        for src, docs in (("gt", gt_train), ("tess", tess_train)):
            held = [d for d in docs if fold[d.doc_id] == f]
            parts += extract_all(ext, build_items(held, seed=100 + f, frac=0.5))
        print(f"  pli {f + 1}/{k} : {time.time() - t0:.0f}s")
    joblib.dump(parts, cache, compress=3)
    return parts


def threshold_at_fpr(scores_clean, fpr):
    return float(np.quantile(scores_clean, 1 - fpr))


def evaluate(scores, types, thr_med, thr_high, rules=None, rare=None):
    y = (types != "clean").astype(int)
    out = {"n_clean": int((y == 0).sum()), "n_fraud": int(y.sum()),
           "roc_auc": float(roc_auc_score(y, scores)), "pr_auc": float(average_precision_score(y, scores))}
    for name, thr in (("medium", thr_med), ("high", thr_high)):
        pred = scores >= thr
        if name == "medium" and rare is not None:
            pred = pred | rare  # politique d'alerte : score >= seuil OU règle rarissime
        out[f"fpr@{name}"] = float(pred[y == 0].mean())
        out[f"recall@{name}"] = float(pred[y == 1].mean())
        out[f"recall_by_type@{name}"] = {t: float(pred[types == t].mean()) for t in sorted(set(types)) if t != "clean"}
    if rules is not None:
        pred = rules >= 1
        # comparaison à taux de faux positifs égal : seuil du score calé sur le FPR des règles
        fpr_r = float(pred[y == 0].mean())
        thr_r = threshold_at_fpr(scores[y == 0], fpr_r)
        out["model_at_rules_fpr"] = {"fpr": float((scores[y == 0] >= thr_r).mean()),
                                     "recall": float((scores[y == 1] >= thr_r).mean())}
        out["rules_only"] = {"fpr": fpr_r, "recall": float(pred[y == 1].mean()),
                             "recall_by_type": {t: float(pred[types == t].mean()) for t in sorted(set(types)) if t != "clean"}}
    return out


def main():
    gt_train, gt_test = load_docs(P / "gt_train.jsonl"), load_docs(P / "gt_test.jsonl")
    tess_train, tess_test = load_docs(P / "tess_train.jsonl"), load_docs(P / "tess_test.jsonl")

    print("Extraction hors pli des reçus d'entraînement...")
    Xtr, Rtr, Ttr, Htr = featurize(oof_train_extractions(gt_train, tess_train))
    ytr = (Ttr != "clean").astype(int)
    print(f"train : {int((ytr == 0).sum())} authentiques, {int(ytr.sum())} falsifiés")

    model = RiskModel().fit(Xtr[ytr == 0], Xtr, ytr)
    s_tr = model.predict_proba(Xtr) * 100
    thr_med = threshold_at_fpr(s_tr[ytr == 0], FPR_MEDIUM)
    thr_high = threshold_at_fpr(s_tr[ytr == 0], FPR_HIGH)
    print(f"seuils : moyen >= {thr_med:.1f}, élevé >= {thr_high:.1f}")

    # Fréquence de déclenchement de chaque règle sur les reçus authentiques (train) : bruit de fond.
    rule_fp = {r: float(np.mean([h[r] for h, t in zip(Htr, Ttr) if t == "clean"])) for r in RULES}

    ext = joblib.load(ROOT / "models" / "extractor.joblib")
    results = {"ref_date": REF_DATE.isoformat(), "thresholds": {"medium": thr_med, "high": thr_high},
               "rule_trigger_rate_on_clean_train": rule_fp}
    curves = {}
    test_types = None
    for src, docs in (("gt", gt_test), ("tess", tess_test)):
        items = build_items(docs, seed=1, frac=1.0)
        X, R, T, H = featurize(extract_all(ext, items))
        test_types = T if src == "gt" else test_types
        s = model.predict_proba(X) * 100
        res = evaluate(s, T, thr_med, thr_high, R, model.rare_rule(X))
        yb = (T != "clean").astype(int)
        res["component_roc_auc"] = {
            "iforest_only": float(roc_auc_score(yb, model.anomaly(X))),
            "rule_pvalues_only": float(roc_auc_score(yb, model.rule_score(X))),
            "full_model": res["roc_auc"]}
        # Généralisation à un type de fraude jamais vu
        loto = {}
        for t in GENERATORS:
            keep = Ttr != t
            m = RiskModel().fit(Xtr[keep & (ytr == 0)], Xtr[keep], ytr[keep])
            st = m.predict_proba(Xtr[keep]) * 100
            th = threshold_at_fpr(st[ytr[keep] == 0], FPR_MEDIUM)
            sc = m.predict_proba(X) * 100
            mask = (T == t) | (T == "clean")
            yy = (T[mask] != "clean").astype(int)
            flag = (sc >= th) | m.rare_rule(X)
            loto[t] = {"roc_auc": float(roc_auc_score(yy, sc[mask])) if yy.any() else None,
                       "recall@medium": float(flag[T == t].mean()) if (T == t).any() else None,
                       "fpr@medium": float(flag[T == "clean"].mean())}
        res["leave_one_type_out"] = loto
        results[src] = res
        curves[src] = (T, s, R)
        print(f"\n[{src}] ROC-AUC {res['roc_auc']:.3f}  PR-AUC {res['pr_auc']:.3f}  "
              f"FPR@moyen {res['fpr@medium']:.1%}  rappel@moyen {res['recall@medium']:.1%}  "
              f"| règles seules : FPR {res['rules_only']['fpr']:.1%} rappel {res['rules_only']['recall']:.1%}"
              f" | modèle au même FPR : rappel {res['model_at_rules_fpr']['recall']:.1%}")
        for t in sorted(res["recall_by_type@medium"]):
            print(f"    {t:18s} rappel {res['recall_by_type@medium'][t]:6.1%}   "
                  f"(règles {res['rules_only']['recall_by_type'][t]:6.1%}, type jamais vu {loto[t]['recall@medium']:6.1%})")

    # coefficients (interprétabilité)
    names = [f"{r}:{k}" for r in RULES for k in ("hit", "resid", "applicable")] + [
        "n_hard_hits", "n_soft_hits", "any_date_rule", "any_missing_field", "n_arith_hits", "max_arith_resid",
        "log_total", "log_max_amount", "total_over_max", "n_items", "n_amounts", "n_lines", "ocr_conf",
        "conf_company", "conf_date", "conf_total", "conf_address", "total_not_mult_005", "anomaly_score", "rule_surprise_max", "rule_surprise_fisher"]
    coefs = dict(sorted(zip(names, model.clf.coef_[0].tolist()), key=lambda kv: -abs(kv[1]))[:15])
    results["top_coefficients"] = coefs

    (ROOT / "reports" / "fraud_metrics.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    joblib.dump({"model": model, "thresholds": {"medium": thr_med, "high": thr_high}, "ref_date": REF_DATE},
                ROOT / "models" / "risk_model.joblib", compress=3)
    np.savez(ROOT / "reports" / "fraud_scores.npz",
             **{f"{k}_{n}": v for k, (T, s, R) in curves.items() for n, v in (("types", T), ("scores", s), ("rules", R))})
    print("\nModèle sauvegardé : models/risk_model.joblib")


if __name__ == "__main__":
    main()
