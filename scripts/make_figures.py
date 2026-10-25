"""Figures du README à partir des métriques sauvegardées (reports/*.json)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
FIG = R / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def grouped_bars(ax, cats, series, colors, fmt="{:.2f}"):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(cats))
    for k, ((name, vals), c) in enumerate(zip(series.items(), colors)):
        pos = x - 0.4 + w * (k + 0.5)
        bars = ax.bar(pos, vals, w - 0.03, color=c, label=name, edgecolor=SURFACE, linewidth=1)
        if k == n - 1 or n <= 2:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, fmt.format(v), ha="center",
                        va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", length=0)


def extraction_fig():
    m = json.loads((R / "extraction_metrics.json").read_text())
    fields = ["company", "date", "address", "total"]
    labels = ["Commerçant", "Date", "Adresse", "Total"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, inp, title in zip(axes, ["gt", "tess"], ["Texte des annotations (OCR parfait)", "OCR Tesseract sur l'image"]):
        series = {
            "Baseline règles": [m[f"rules@{inp}"][f]["f1"] for f in fields],
            "Modèle (texte + mise en page)": [m[f"ml_mix@{inp}"][f]["f1"] for f in fields],
            "Plafond atteignable": [m[f"oracle@{inp}"][f]["f1"] for f in fields],
        }
        grouped_bars(ax, labels, series, [S2, S1, "#c9c8c2"])
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("F1 (correspondance exacte)")
    fig.suptitle("Extraction des champs : F1 par champ sur 125 reçus de test", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9)
    fig.savefig(FIG / "extraction_f1.png", dpi=160, bbox_inches="tight")


TYPE_FR = {
    "date_future": "Date future", "date_impossible": "Date impossible", "field_removed": "Champ supprimé",
    "item_altered": "Article modifié", "tax_altered": "Taxe modifiée", "total_inflated": "Total gonflé",
    "total_consistent": "Total modifié\n(cohérent)",
}


def fraud_fig():
    m = json.loads((R / "fraud_metrics.json").read_text())["gt"]
    types = sorted(m["recall_by_type@medium"], key=lambda t: -m["recall_by_type@medium"][t])
    series = {
        f"Règles seules (FPR {m['rules_only']['fpr']:.0%})": [m["rules_only"]["recall_by_type"][t] for t in types],
        f"Score de risque, seuil moyen (FPR {m['fpr@medium']:.0%})": [m["recall_by_type@medium"][t] for t in types],
        "Score, type jamais vu à l'entraînement": [m["leave_one_type_out"][t]["recall@medium"] for t in types],
    }
    fig, ax = plt.subplots(figsize=(10, 3.8))
    grouped_bars(ax, [TYPE_FR[t] for t in types], series, [S2, S1, S3], fmt="{:.0%}")
    ax.set_ylabel("Rappel (part des fraudes signalées)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    ax.set_title("Détection par type de fraude synthétique (reçus de test, texte des annotations)",
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fraud_recall_by_type.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    extraction_fig()
    fraud_fig()
    print("figures :", sorted(p.name for p in FIG.glob("*.png")))
