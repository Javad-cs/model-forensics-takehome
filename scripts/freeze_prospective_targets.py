import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ORDER = ROOT / "results/structural_selection_order_DO_NOT_OPEN_UNTIL_LABELS_LOCKED.json"
SOURCE = (
    ROOT
    / "repos/value_leakage_data/static/data/giraffes/qwen3.5-35"
    / "v1_giraffes_accurate.json"
)
OUT = ROOT / "results/prospective_active_decision_targets_frozen.json"

SELECTED = [
    "B008",
    "B011",
    "B012",
    "B028",
    "B035",
    "B036",
    "B039",
    "B047",
]

if OUT.exists():
    raise SystemExit(f"{OUT} already exists; refusing to overwrite.")

with ORDER.open(encoding="utf-8") as f:
    order = json.load(f)

with SOURCE.open(encoding="utf-8") as f:
    data = json.load(f)

mapping = {x["blind_id"]: x for x in order["order"]}

assert set(SELECTED).issubset(mapping)

targets = []

for blind_id in SELECTED:
    m = mapping[blind_id]

    row_idx = int(m["source_row"])
    line_idx = int(m["source_line_index"])

    row = data["rows"][row_idx]
    lines = row["reasoning"].splitlines(keepends=True)

    assert 0 <= line_idx < len(lines)

    S = lines[line_idx]
    P = "".join(lines[:line_idx])

    targets.append({
        "blind_id": blind_id,
        "review_index": int(m["review_index"]),
        "source_row_index": row_idx,
        "source_line_index": line_idx,
        "direction": row["direction"],
        "relative_position": m["relative_position"],
        "remaining_chars": m["remaining_chars"],
        "prefix_reasoning": P,
        "correction_episode": S,

        # Fixed before regeneration.
        "regeneration_samples": 10,
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 160,
        },
    })

frozen = {
    "selection_status": (
        "Prospectively selected by frozen blinded structural-review "
        "procedure; exactly 8 ACTIVE_DECISION targets."
    ),
    "selected_blind_ids": SELECTED,
    "n_targets": len(targets),
    "natural_counterfactual_usability_rule": (
        "Target is naturally branchable iff at least 4/10 screened "
        "first reasoning units are classified NOT_REGENERATED. "
        "AMBIGUOUS does not count as NOT_REGENERATED."
    ),
    "important_analysis_rule": (
        "All 8 targets remain reported regardless of regeneration rate "
        "or downstream causal result. Regeneration determines only "
        "whether the natural sentence-resampling causal comparison is "
        "identifiable at that target."
    ),
    "targets": targets,
}

with OUT.open("w", encoding="utf-8") as f:
    json.dump(frozen, f, ensure_ascii=False, indent=2)

print("WROTE:", OUT)
print("N TARGETS:", len(targets))

print("\nUNBLINDED SELECTED TARGETS ONLY:")
for t in targets:
    print(
        t["blind_id"],
        "| row", t["source_row_index"],
        "|", t["direction"],
        "| line", t["source_line_index"],
    )
    print("  S:", t["correction_episode"].strip())

