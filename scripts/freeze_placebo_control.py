import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CAUSAL = ROOT / "results/causal_experiment_frozen.json"

OUT = ROOT / "results/placebo_experiment_frozen.json"
DESIGN = ROOT / "results/placebo_experiment_design_frozen.txt"

for p in [OUT, DESIGN]:
    if p.exists():
        raise SystemExit(f"{p} already exists; refusing to overwrite.")

with CAUSAL.open(encoding="utf-8") as f:
    causal = json.load(f)

TARGETS = [
    "B008",
    "B011",
    "B012",
    "B036",
    "B039",
    "B047",
]

by_id = {
    t["blind_id"]: t
    for t in causal["targets"]
}

placebo_targets = []

for tid in TARGETS:

    t = by_id[tid]

    # Frozen natural alternatives, sorted only by their
    # pre-existing source regeneration seed.
    alternatives = sorted(
        t["pairs"],
        key=lambda x: int(x["source_regeneration_seed"]),
    )

    k = len(alternatives)
    assert k >= 4

    placebo_pairs = []

    for i, left in enumerate(alternatives):

        right = alternatives[(i + 1) % k]

        left_seed = int(left["source_regeneration_seed"])
        right_seed = int(right["source_regeneration_seed"])

        # Fresh seed, disjoint from causal-generation seeds.
        continuation_seed = 500000 + left_seed

        placebo_pairs.append({
            "pair_index": i + 1,

            "left_source_regeneration_seed":
                left_seed,

            "right_source_regeneration_seed":
                right_seed,

            "left_S_prime":
                left["natural_alternative_S_prime"],

            "right_S_prime":
                right["natural_alternative_S_prime"],

            "continuation_seed":
                continuation_seed,
        })

    placebo_targets.append({
        "blind_id": tid,
        "source_row_index": t["source_row_index"],
        "source_line_index": t["source_line_index"],
        "direction": t["direction"],
        "prefix_reasoning": t["prefix_reasoning"],
        "n_placebo_pairs": len(placebo_pairs),
        "pairs": placebo_pairs,
    })


n_pairs = sum(
    t["n_placebo_pairs"]
    for t in placebo_targets
)

assert n_pairs == 41
assert 2 * n_pairs == 82


manifest = {
    "status": "FROZEN BEFORE PLACEBO GENERATION",

    "purpose": (
        "Estimate generic trajectory sensitivity to replacing one "
        "naturally sampled non-correction thought with another "
        "naturally sampled non-correction thought at the same "
        "intervention point."
    ),

    "included_targets": TARGETS,

    "pairing_rule": (
        "Within each target, sort all frozen NOT_REGENERATED natural "
        "alternatives by source_regeneration_seed. Pair each alternative "
        "with the next alternative in cyclic order, including last->first. "
        "No semantic content or downstream outcome is used for pairing."
    ),

    "n_independent_targets": 6,
    "n_placebo_pairs": n_pairs,
    "n_total_generations": 2 * n_pairs,

    "sampling": {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": 8192,
    },

    "analysis_unit": (
        "Target/prefix remains the independent forensic unit. "
        "The 41 placebo pairs estimate within-target generic "
        "branch sensitivity."
    ),

    "precommitted_control_analysis": {
        "primary": (
            "For each target, compute the mean absolute signed-margin "
            "difference between the two natural branches: "
            "G_i = mean_j |M(Y_right)-M(Y_left)|."
        ),

        "comparison": (
            "Compare G_i with the magnitude and consistency of the "
            "already-frozen correction-vs-natural effects. "
            "The placebo orientation is arbitrary and therefore its "
            "signed mean is not treated as a morally meaningful effect."
        ),

        "interpretation": (
            "If correction-vs-natural effects, especially B012, are "
            "large and directionally consistent relative to natural-vs-natural "
            "variation, this supports correction-specific causal content. "
            "If natural-vs-natural perturbations are similarly large, "
            "interpret the original effect more cautiously as generic "
            "trajectory sensitivity."
        ),
    },

    "targets": placebo_targets,
}


with OUT.open("w", encoding="utf-8") as f:
    json.dump(
        manifest,
        f,
        ensure_ascii=False,
        indent=2,
    )


design_text = """NATURAL-vs-NATURAL PLACEBO CONTROL

Frozen before any placebo continuation is generated.

PURPOSE
Test whether the original correction effect is specific to the correction
content, or whether replacing any natural thought at the same position produces
similarly large trajectory changes.

TARGETS
Exactly the same six frozen causal targets:

B008, B011, B012, B036, B039, B047.

No new targets are searched for.

PLACEBO INTERVENTION
All previously accepted NOT_REGENERATED natural alternatives S' are used.

Within each target:
1. sort alternatives by frozen source_regeneration_seed;
2. pair each S'_i with the next S'_(i+1);
3. use cyclic wraparound for the final alternative.

No semantic content or downstream outcome is used to choose pairs.

Each placebo pair compares:

    P + S'_left

versus

    P + S'_right

using the same fresh continuation seed.

COUNTS
41 placebo pairs.
82 total placebo continuations.
6 independent targets.

SAMPLING
temperature = 1.0
top_p = 1.0
top_k = 0
max_new_tokens = 8192

PRIMARY CONTROL METRIC
For target i:

    G_i = mean_j | M(Y_right) - M(Y_left) |

where

    M(Y) = d * (Y - T) / T

and T = 20,200,000.

Because left/right orientation is arbitrary, signed placebo direction is not
interpreted as morally meaningful.

INTERPRETATION
The placebo estimates generic branch sensitivity.

If correction-vs-natural effects are substantially more systematic than this
natural-vs-natural variation, that supports a correction-specific causal role.

If placebo perturbations are comparably large, the original intervention
should be interpreted more cautiously as generic trajectory sensitivity.

The target/prefix remains the independent analysis unit.
"""

DESIGN.write_text(
    design_text,
    encoding="utf-8",
)


print("FROZEN PLACEBO TARGETS:")

for t in placebo_targets:
    pairs = [
        (
            p["left_source_regeneration_seed"],
            p["right_source_regeneration_seed"],
        )
        for p in t["pairs"]
    ]

    print(
        t["blind_id"],
        "|", t["direction"],
        "| pairs =", t["n_placebo_pairs"],
        "|", pairs,
    )

print()
print("Independent targets:", len(placebo_targets))
print("Placebo pairs:", n_pairs)
print("Total generations:", 2 * n_pairs)

print()
print("WROTE:")
print(OUT)
print(DESIGN)
