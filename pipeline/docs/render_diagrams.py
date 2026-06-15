"""Génère deux diagrammes PNG (architecture + flux) avec matplotlib.

Sorties : docs/architecture_overview.png et docs/data_flow.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PALETTE = {
    "bronze": "#F4D6A7",
    "silver": "#D6DDE0",
    "ml": "#BFD9F5",
    "orch": "#FFE89A",
    "code": "#E4D7F2",
    "edge": "#2C3E50",
}


def box(ax, x, y, w, h, text, color, fontsize=9, fontweight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=PALETTE["edge"],
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight, color="#1b1b1b",
    )
    return (x, y, w, h)


def arrow(ax, src, dst, label=None, label_color="#2c3e50", offset=0.0):
    """Flèche entre deux boîtes (centres-bords)."""
    sx = src[0] + src[2] / 2
    sy = src[1] + src[3] / 2
    dx = dst[0] + dst[2] / 2
    dy = dst[1] + dst[3] / 2

    if abs(dx - sx) > abs(dy - sy):
        # horizontal
        sx = src[0] + src[2] if dx > sx else src[0]
        dx = dst[0] if dx > sx else dst[0] + dst[2]
    else:
        sy = src[1] + src[3] if dy > sy else src[1]
        dy = dst[1] if dy > sy else dst[1] + dst[3]

    arr = FancyArrowPatch(
        (sx, sy), (dx, dy),
        arrowstyle="->,head_length=8,head_width=5",
        linewidth=1.2,
        color=PALETTE["edge"],
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)

    if label:
        ax.text(
            (sx + dx) / 2, (sy + dy) / 2 + offset, label,
            ha="center", va="center",
            fontsize=7.5, color=label_color,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
        )


# ---------------------------------------------------------------------------
# Diagramme 1 : architecture d'ensemble
# ---------------------------------------------------------------------------

def render_overview() -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_axis_off()
    ax.set_title(
        "Architecture CIB Forecast — CSV → HDFS → PySpark → ML → Airflow",
        fontsize=14, fontweight="bold", pad=14,
    )

    # Colonnes (Bronze / Silver / ML)
    ax.add_patch(mpatches.Rectangle((0.4, 1.2), 3.2, 6.5, fill=True, alpha=0.18, color=PALETTE["bronze"], lw=0))
    ax.text(2.0, 7.5, "BRONZE (cib_bronze)", ha="center", fontsize=10, fontweight="bold")
    ax.add_patch(mpatches.Rectangle((4.2, 1.2), 4.4, 6.5, fill=True, alpha=0.18, color=PALETTE["silver"], lw=0))
    ax.text(6.4, 7.5, "SILVER (cib_silver)", ha="center", fontsize=10, fontweight="bold")
    ax.add_patch(mpatches.Rectangle((9.2, 1.2), 4.4, 6.5, fill=True, alpha=0.18, color=PALETTE["ml"], lw=0))
    ax.text(11.4, 7.5, "ML (cib_ml)", ha="center", fontsize=10, fontweight="bold")

    # Bronze
    trx = box(ax, 0.7, 5.7, 2.6, 0.9, "transactions_raw\n(HDFS / CSV ou ORC)", PALETTE["bronze"], 9)
    cal = box(ax, 0.7, 4.5, 2.6, 0.9, "calendar_weekly_flags", PALETTE["bronze"], 9)
    mac = box(ax, 0.7, 3.3, 2.6, 0.9, "macro_indicators_weekly", PALETTE["bronze"], 9)
    syn = box(ax, 0.7, 1.5, 2.6, 0.9, "Datagen synthétique\n(scripts/bootstrap_bronze.py)", PALETTE["code"], 8.5)
    arrow(ax, syn, trx)
    arrow(ax, syn, cal)
    arrow(ax, syn, mac)

    # Silver
    wk = box(ax, 4.5, 5.7, 3.8, 0.9, "weekly_cashflow_account\n(step 1)", PALETTE["silver"], 9)
    qm = box(ax, 4.5, 4.5, 3.8, 0.9, "account_quality_metrics\n(steps 2-7)", PALETTE["silver"], 9)
    pol = box(ax, 4.5, 3.3, 3.8, 0.9, "account_policy\n(use_externals_*)", PALETTE["silver"], 9)
    arrow(ax, trx, wk, label="transformation_job.py")
    arrow(ax, trx, qm, label="transformation_job.py", offset=-0.18)
    arrow(ax, qm, pol, label="policy_job.py")

    # ML
    feat = box(ax, 9.5, 5.7, 3.8, 0.9, "features_cib_forecast\n(ORC, step 8)", PALETTE["ml"], 9)
    mods = box(ax, 9.5, 4.5, 3.8, 0.9, "Modèles : RF (régression)\n+ Logistique (classification)", PALETTE["ml"], 8.5)
    pred = box(ax, 9.5, 3.3, 3.8, 0.9, "predictions_cib_forecast", PALETTE["ml"], 9)
    arrow(ax, wk, feat, label="features.py")
    arrow(ax, cal, feat, label="join")
    arrow(ax, mac, feat, label="join (+1 sem)", offset=0.18)
    arrow(ax, pol, feat, label="filtre comptes", offset=-0.18)
    arrow(ax, feat, mods, label="train.py")
    arrow(ax, mods, pred, label="inference.py")

    # Airflow strip
    af = box(ax, 0.7, 0.3, 12.6, 0.8, "Airflow DAG  •  bootstrap → transform → policy → features → train → inference", PALETTE["orch"], 9.5, "bold")
    ax.annotate("", xy=(1.8, 1.2), xytext=(1.8, 1.05),
                arrowprops=dict(arrowstyle="->", color=PALETTE["edge"]))
    ax.text(7.0, 1.1, "(orchestration)", ha="center", fontsize=8, color="#555")

    out = OUT / "architecture_overview.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Diagramme 2 : steps du notebook → tables
# ---------------------------------------------------------------------------

def render_steps() -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.set_axis_off()
    ax.set_title("Flux des étapes (notebook → tables silver / ML)", fontsize=14, fontweight="bold", pad=14)

    s1 = box(ax, 0.3, 3.5, 1.7, 0.8, "Step 1\nAgg hebdo", PALETTE["bronze"])
    s2 = box(ax, 2.3, 3.5, 1.7, 0.8, "Step 2\nStats globales", PALETTE["silver"])
    s3 = box(ax, 4.3, 3.5, 1.7, 0.8, "Step 3\nEligibility", PALETTE["silver"])
    s4 = box(ax, 6.3, 3.5, 1.7, 0.8, "Step 4\nADF / ACF / PACF", PALETTE["silver"])
    s5 = box(ax, 8.3, 3.5, 1.7, 0.8, "Step 5\nValidité (NaN)", PALETTE["silver"])
    s6 = box(ax, 10.3, 3.5, 1.7, 0.8, "Step 6\nScores composites", PALETTE["silver"])
    s7 = box(ax, 12.3, 3.5, 1.5, 0.8, "Step 7\nFiltre qualité", PALETTE["silver"])

    for a, b in [(s1, s2), (s2, s3), (s3, s4), (s4, s5), (s5, s6), (s6, s7)]:
        arrow(ax, a, b)

    s8 = box(ax, 4.3, 1.7, 5.5, 0.9, "Step 8 — Features ML\n(lags 4 sem, rolling, joins calendar + macro+1w)", PALETTE["ml"])
    train = box(ax, 10.5, 1.7, 3.3, 0.9, "Train + Inference\n(RF + Logistique)", PALETTE["ml"])

    arrow(ax, s1, s8, label="weekly")
    arrow(ax, s7, s8, label="comptes qualité")
    arrow(ax, s8, train, label="X, y")

    ax.text(7.0, 0.7, "Politique d'activation des externes (account_policy) appliquée à chaque compte",
            ha="center", fontsize=9, color="#444")

    out = OUT / "data_flow.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = render_overview()
    p2 = render_steps()
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")
