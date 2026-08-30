import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "results/qwen35_self_correction_candidates.csv"

ORDER_OUT = (
    ROOT
    / "results/structural_selection_order_DO_NOT_OPEN_UNTIL_LABELS_LOCKED.json"
)

BLIND_OUT = ROOT / "results/structural_review_batch1_blind.csv"

SEED = 20260828

# Everything we had manually inspected before this prospective stage.
PREVIOUSLY_SEEN_ROWS = {
    164, 178, 106, 14,
    79, 208, 29, 9,
    60, 17, 102, 166,
}

for p in [ORDER_OUT, BLIND_OUT]:
    if p.exists():
        raise SystemExit(f"{p} already exists; refusing to overwrite.")


def as_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


def mask_numbers(text):
    # Preserve the existence of numerical reasoning while hiding values.
    pattern = (
        r"(?<![\w])"
        r"[-+]?"
        r"(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
        r"(?:\s*(?:%|[kKmMbB]))?"
        r"(?![\w])"
    )
    return re.sub(pattern, "<NUM>", text or "")


def mask_valence(text, flip):
    """
    Hide good/bad identity.

    Mapping is randomized independently for each candidate so CAUSE_A
    cannot be interpreted as 'good' across the sheet.
    """
    text = text or ""

    if flip:
        good_token = "VALENCE_B"
        bad_token = "VALENCE_A"
    else:
        good_token = "VALENCE_A"
        bad_token = "VALENCE_B"

    # Broad masking is intentional: structural classification does not
    # require knowing moral valence.
    text = re.sub(r"\bgood\b", good_token, text, flags=re.I)
    text = re.sub(r"\bbad\b", bad_token, text, flags=re.I)

    return text


def blind_text(text, flip):
    return mask_numbers(mask_valence(text, flip))


# ------------------------------------------------------------
# 1. Apply ONLY the frozen automated eligibility criteria.
# ------------------------------------------------------------
eligible_events = []

with SRC.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for r in reader:
        if r["question"].strip().lower() != "giraffes":
            continue

        if not as_bool(r["primary_candidate"]):
            continue

        pos = float(r["relative_position"])
        rem = int(float(r["remaining_chars"]))
        row = int(r["row"])
        line = int(r["line_index"])

        if not (0.80 <= pos <= 0.95):
            continue

        if not (800 <= rem <= 4500):
            continue

        if row in PREVIOUSLY_SEEN_ROWS:
            continue

        eligible_events.append({
            "row": row,
            "line_index": line,
            "relative_position": pos,
            "remaining_chars": rem,

            # Content is stored for producing the blind sheet,
            # but is NOT used for selection/order.
            "candidate_sentence": r["candidate_sentence"],
            "context_before": r["context_before"],
            "context_after": r["context_after"],
        })


# ------------------------------------------------------------
# 2. One candidate per independent rollout.
#    Frozen rule: FIRST eligible event in reasoning order.
# ------------------------------------------------------------
by_row = {}

for x in eligible_events:
    row = x["row"]

    if (
        row not in by_row
        or x["line_index"] < by_row[row]["line_index"]
    ):
        by_row[row] = x

candidates = list(by_row.values())

# Deterministic state before random permutation.
candidates.sort(key=lambda x: (x["row"], x["line_index"]))


# ------------------------------------------------------------
# 3. Frozen random permutation.
# ------------------------------------------------------------
rng = random.Random(SEED)
rng.shuffle(candidates)


# ------------------------------------------------------------
# 4. Give candidates anonymous IDs.
#
# Cause-valence masking gets an independently sampled flip.
# The flip is stored only in the sealed mapping, not the blind sheet.
# ------------------------------------------------------------
sealed = []

blind_rows = []

for i, x in enumerate(candidates, start=1):
    blind_id = f"B{i:03d}"
    flip = bool(rng.getrandbits(1))

    sealed.append({
        "review_index": i,
        "blind_id": blind_id,
        "source_row": x["row"],
        "source_line_index": x["line_index"],
        "relative_position": x["relative_position"],
        "remaining_chars": x["remaining_chars"],
        "valence_mask_flip": flip,
    })

    # Only first 20 are exposed in batch 1.
    if i <= 20:
        before = x["context_before"] or ""
        after = x["context_after"] or ""

        # Keep only genuinely local context.
        before = before[-1600:]
        after = after[:1600]

        blind_rows.append({
            "review_index": i,
            "blind_id": blind_id,
            "candidate_correction": blind_text(
                x["candidate_sentence"], flip
            ),
            "context_before": blind_text(before, flip),
            "context_after": blind_text(after, flip),

            # Intentionally blank for manual prospective labeling.
            "structural_label": "",
            "brief_reason": "",
        })


# ------------------------------------------------------------
# 5. Save sealed order.
#    DO NOT OPEN until structural labels are locked.
# ------------------------------------------------------------
with ORDER_OUT.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "seed": SEED,
            "previously_seen_rows_excluded": sorted(
                PREVIOUSLY_SEEN_ROWS
            ),
            "automated_filter": {
                "question": "giraffes",
                "primary_candidate": True,
                "relative_position": [0.80, 0.95],
                "remaining_chars": [800, 4500],
                "within_rollout_choice":
                    "smallest line_index among eligible events",
            },
            "n_eligible_events_after_seen_exclusion":
                len(eligible_events),
            "n_unique_rollouts": len(candidates),
            "order": sealed,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )


# ------------------------------------------------------------
# 6. Save blind batch.
# ------------------------------------------------------------
with BLIND_OUT.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "review_index",
            "blind_id",
            "candidate_correction",
            "context_before",
            "context_after",
            "structural_label",
            "brief_reason",
        ],
    )

    writer.writeheader()
    writer.writerows(blind_rows)


print("Eligible candidate events:", len(eligible_events))
print("Independent prospective rollouts:", len(candidates))
print("Blind batch size:", len(blind_rows))

print("\nWROTE:")
print(ORDER_OUT)
print(BLIND_OUT)

print(
    "\nIMPORTANT: Do NOT open the sealed order file "
    "until structural labels are locked."
)
