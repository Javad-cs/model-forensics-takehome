import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAIR_FILE = ROOT / "results/processed/causal_pair_effects.csv"

OUTDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_ORDER = [
    "B008",
    "B011",
    "B012",
    "B036",
    "B039",
    "B047",
]


# ============================================================
# Load pair-level causal effects
# ============================================================

rows = []

with PAIR_FILE.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for r in reader:
        rows.append({
            "target": r["target"],
            "direction": r["direction"],
            "source_seed": int(r["source_regeneration_seed"]),
            "B_final_pct": 100.0 * float(r["pair_B_final"]),
            "B_next_pct": 100.0 * float(r["pair_B_next"]),
        })


assert len(rows) == 41

by_target = defaultdict(list)

for r in rows:
    by_target[r["target"]].append(r)

for tid in TARGET_ORDER:
    assert tid in by_target


# ============================================================
# Target-level summaries
# ============================================================

summary = {}

for tid in TARGET_ORDER:

    vals = [
        r["B_final_pct"]
        for r in by_target[tid]
    ]

    mean_effect = statistics.mean(vals)
    median_effect = statistics.median(vals)

    loo_means = []

    for i in range(len(vals)):
        remaining = vals[:i] + vals[i + 1:]
        loo_means.append(
            statistics.mean(remaining)
        )

    summary[tid] = {
        "mean": mean_effect,
        "median": median_effect,
        "loo_min": min(loo_means),
        "loo_max": max(loo_means),
        "positive": sum(x > 1e-12 for x in vals),
        "negative": sum(x < -1e-12 for x in vals),
        "zero": sum(abs(x) <= 1e-12 for x in vals),
    }


# ============================================================
# Save exact summary data used in figure
# ============================================================

SUMMARY_FILE = OUTDIR / "causal_figure_summary.csv"

with SUMMARY_FILE.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    fieldnames = [
        "target",
        "n_pairs",
        "mean_B_final_pct_threshold",
        "median_pair_B_final_pct_threshold",
        "positive_pairs",
        "negative_pairs",
        "zero_pairs",
        "LOO_min_mean_B_final_pct_threshold",
        "LOO_max_mean_B_final_pct_threshold",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for tid in TARGET_ORDER:
        s = summary[tid]

        writer.writerow({
            "target": tid,
            "n_pairs": len(by_target[tid]),
            "mean_B_final_pct_threshold": s["mean"],
            "median_pair_B_final_pct_threshold": s["median"],
            "positive_pairs": s["positive"],
            "negative_pairs": s["negative"],
            "zero_pairs": s["zero"],
            "LOO_min_mean_B_final_pct_threshold":
                s["loo_min"],
            "LOO_max_mean_B_final_pct_threshold":
                s["loo_max"],
        })


# ============================================================
# Deterministic vertical jitter
#
# Depends only on frozen source seed, never causal outcome.
# ============================================================

def deterministic_jitter(seed):
    x = ((seed * 37) % 101) / 100.0
    return (x - 0.5) * 0.30


# ============================================================
# Plot
# ============================================================

def make_figure(
    path_stem,
    xlim=None,
    clipped_note=False,
):

    fig, ax = plt.subplots(
        figsize=(8.2, 4.6)
    )

    # Zero causal effect
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.2,
        zorder=0,
    )

    ypos = list(range(len(TARGET_ORDER)))

    # --------------------------------------------------------
    # Individual matched pairs
    # --------------------------------------------------------

    all_x = []
    all_y = []

    for y, tid in enumerate(TARGET_ORDER):

        for r in by_target[tid]:

            all_x.append(
                r["B_final_pct"]
            )

            all_y.append(
                y
                + deterministic_jitter(
                    r["source_seed"]
                )
            )

    pair_handle = ax.scatter(
        all_x,
        all_y,
        s=38,
        alpha=0.55,
        label="Matched pair",
        zorder=2,
    )

    # --------------------------------------------------------
    # Leave-one-pair-out target-mean ranges
    #
    # All ranges intentionally share the same visual style.
    # --------------------------------------------------------

    loo_handle = None
    loo_color = None

    for i, tid in enumerate(TARGET_ORDER):

        line, = ax.plot(
            [
                summary[tid]["loo_min"],
                summary[tid]["loo_max"],
            ],
            [i, i],
            linewidth=3,
            alpha=0.75,
            zorder=1,
        )

        if loo_handle is None:
            loo_handle = line
            loo_color = line.get_color()

        else:
            line.set_color(
                loo_color
            )

    loo_handle.set_label(
        "Leave-one-pair-out mean range"
    )

    # --------------------------------------------------------
    # Target means
    # --------------------------------------------------------

    means = [
        summary[t]["mean"]
        for t in TARGET_ORDER
    ]

    mean_handle = ax.scatter(
        means,
        ypos,
        marker="D",
        s=70,
        label="Target mean",
        zorder=4,
    )

    # --------------------------------------------------------
    # Axes / labels
    # --------------------------------------------------------

    ax.set_yticks(
        ypos,
        TARGET_ORDER,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Correction benefit, final estimate (% of threshold)"
    )

    ax.set_ylabel(
        "Intervention target"
    )

    ax.set_title(
        "Causal effect of verbalized self-correction"
    )

    if xlim is not None:
        ax.set_xlim(*xlim)

    if clipped_note:
        ax.text(
            0.99,
            0.02,
            "Points outside displayed range are clipped",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    ax.legend(
        handles=[
            pair_handle,
            loo_handle,
            mean_handle,
        ],
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        path_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        path_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main full-range figure
# ============================================================

make_figure(
    OUTDIR / "causal_pair_effects_full"
)


# ============================================================
# Secondary zoomed view
#
# SAME data. No observations removed from analysis.
# Some points are outside the displayed x-range.
# ============================================================

make_figure(
    OUTDIR / "causal_pair_effects_zoom",
    xlim=(-10, 10),
    clipped_note=True,
)


# ============================================================
# Audit printout
# ============================================================

print("\nFIGURE DATA")
print("=" * 80)

for tid in TARGET_ORDER:

    s = summary[tid]

    print(
        f"{tid}: "
        f"n={len(by_target[tid])} | "
        f"mean={s['mean']:+.3f}% | "
        f"median={s['median']:+.3f}% | "
        f"+/-/0="
        f"{s['positive']}/"
        f"{s['negative']}/"
        f"{s['zero']} | "
        f"LOO=["
        f"{s['loo_min']:+.3f}, "
        f"{s['loo_max']:+.3f}]%"
    )


print("\nWROTE:")

for p in [
    OUTDIR / "causal_pair_effects_full.pdf",
    OUTDIR / "causal_pair_effects_full.png",
    OUTDIR / "causal_pair_effects_zoom.pdf",
    OUTDIR / "causal_pair_effects_zoom.png",
    SUMMARY_FILE,
]:
    print(p)
