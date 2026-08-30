import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RAW_FILES = [
    ROOT / "results/raw/causal_worker01.jsonl",
    ROOT / "results/raw/causal_worker23.jsonl",
]

LOCKED = ROOT / "labels/causal_outcome_extraction_blind_locked.csv"

OUTCOMES = ROOT / "results/processed/causal_outcomes_unblinded_reproduced.csv"
PAIR_OUT = ROOT / "results/processed/causal_pair_effects_reproduced.csv"
TARGET_OUT = ROOT / "results/processed/causal_target_effects_reproduced.csv"

SHUFFLE_SEED = 20260829
THRESHOLD = 20_200_000


# Load causal generations in their original worker-file order.
records = []

for path in RAW_FILES:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

assert len(records) == 82, f"Expected 82 causal generations, found {len(records)}"


# Reconstruct the blinded extraction ordering.
indices = list(range(len(records)))
rng = random.Random(SHUFFLE_SEED)
rng.shuffle(indices)

mapping = {
    f"Y{i:03d}": original_index
    for i, original_index in enumerate(indices, start=1)
}


# Load outcome values that were locked before unblinding.
locked = {}

with LOCKED.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        extraction_id = row["extraction_id"]

        locked[extraction_id] = {
            "Y_next": float(row["Y_next"]) if row["Y_next"] else None,
            "Y_next_status": row["Y_next_status"],
            "Y_final": float(row["Y_final"]) if row["Y_final"] else None,
            "Y_final_status": row["Y_final_status"],
        }

assert len(locked) == 82, f"Expected 82 locked extractions, found {len(locked)}"


# Reconnect identities only after the blinded extraction is loaded.
unblinded = []

for i in range(1, 83):
    extraction_id = f"Y{i:03d}"

    record = records[mapping[extraction_id]]
    extraction = locked[extraction_id]

    assert extraction["Y_next_status"] == "CLEAR"
    assert extraction["Y_final_status"] == "CLEAR"

    d = +1 if record["direction"] == "above_good" else -1

    y_next = extraction["Y_next"]
    y_final = extraction["Y_final"]

    m_next = d * (y_next - THRESHOLD) / THRESHOLD
    m_final = d * (y_final - THRESHOLD) / THRESHOLD

    unblinded.append({
        "extraction_id": extraction_id,
        "blind_id": record["blind_id"],
        "direction": record["direction"],
        "source_regeneration_seed": int(record["source_regeneration_seed"]),
        "continuation_seed": int(record["continuation_seed"]),
        "arm": record["arm"],
        "Y_next": y_next,
        "Y_final": y_final,
        "M_next": m_next,
        "M_final": m_final,
    })


with OUTCOMES.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(unblinded[0].keys()))
    writer.writeheader()
    writer.writerows(unblinded)


# Pair-level effects.
by_target = defaultdict(list)

for row in unblinded:
    by_target[row["blind_id"]].append(row)

pair_rows = []
target_rows = []

for target in sorted(by_target):
    rows = by_target[target]
    direction = rows[0]["direction"]

    seeds = sorted({
        row["source_regeneration_seed"]
        for row in rows
    })

    for seed in seeds:
        correction = next(
            row for row in rows
            if row["source_regeneration_seed"] == seed
            and row["arm"] == "correction"
        )

        natural = next(
            row for row in rows
            if row["source_regeneration_seed"] == seed
            and row["arm"] == "natural"
        )

        pair_rows.append({
            "target": target,
            "direction": direction,
            "source_regeneration_seed": seed,
            "correction_Y_final": correction["Y_final"],
            "natural_Y_final": natural["Y_final"],
            "pair_B_final": natural["M_final"] - correction["M_final"],
            "correction_Y_next": correction["Y_next"],
            "natural_Y_next": natural["Y_next"],
            "pair_B_next": natural["M_next"] - correction["M_next"],
        })


# Target-level effects.
def mean(values):
    return sum(values) / len(values)


for target in sorted(by_target):
    rows = by_target[target]

    correction = [row for row in rows if row["arm"] == "correction"]
    natural = [row for row in rows if row["arm"] == "natural"]

    assert len(correction) == len(natural)

    target_rows.append({
        "target": target,
        "direction": rows[0]["direction"],
        "n_pairs": len(correction),

        "mean_final_correction": mean([row["Y_final"] for row in correction]),
        "mean_final_natural": mean([row["Y_final"] for row in natural]),

        "B_final":
            mean([row["M_final"] for row in natural])
            - mean([row["M_final"] for row in correction]),

        "mean_next_correction": mean([row["Y_next"] for row in correction]),
        "mean_next_natural": mean([row["Y_next"] for row in natural]),

        "B_next":
            mean([row["M_next"] for row in natural])
            - mean([row["M_next"] for row in correction]),

        "favored_final_rate_correction":
            mean([row["M_final"] > 0 for row in correction]),

        "favored_final_rate_natural":
            mean([row["M_final"] > 0 for row in natural]),
    })


with PAIR_OUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
    writer.writeheader()
    writer.writerows(pair_rows)


with TARGET_OUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(target_rows[0].keys()))
    writer.writeheader()
    writer.writerows(target_rows)


print("Reproduced:")
print(OUTCOMES)
print(PAIR_OUT)
print(TARGET_OUT)

print()
print("Records:", len(unblinded))
print("Pairs:", len(pair_rows))
print("Targets:", len(target_rows))

print()
for row in target_rows:
    print(
        row["target"],
        "| n =", row["n_pairs"],
        "| B_final =", f"{100 * row['B_final']:+.3f}%",
        "| B_next =", f"{100 * row['B_next']:+.3f}%",
    )
