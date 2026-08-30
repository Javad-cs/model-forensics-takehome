import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATHS = [
    ROOT / "results/causal_worker01.jsonl",
    ROOT / "results/causal_worker23.jsonl",
]

OUT = ROOT / "results/causal_trace_audit_all6.txt"

TARGETS = [
    "B008",
    "B011",
    "B012",
    "B036",
    "B039",
    "B047",
]

if OUT.exists():
    raise SystemExit(f"{OUT} already exists; refusing to overwrite.")

rows = []

for p in PATHS:
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

by_pair = defaultdict(dict)

for r in rows:
    tid = r["blind_id"]

    if tid not in TARGETS:
        continue

    key = (
        tid,
        int(r["source_regeneration_seed"]),
    )

    by_pair[key][r["arm"]] = r


def beginning(text, n=1800):
    text = text or ""
    return text[:n]


def ending(text, n=1200):
    text = text or ""
    if len(text) <= n:
        return text
    return text[-n:]


with OUT.open("w", encoding="utf-8") as out:

    out.write(
        "CAUSAL TRACE AUDIT\n"
        "==================\n\n"
        "Exploratory qualitative audit performed AFTER causal "
        "outcomes were estimated.\n"
        "Every matched pair from all six causal targets is shown; "
        "no pair is selected based on effect size or sign.\n\n"
    )

    for tid in TARGETS:

        keys = sorted(
            k for k in by_pair
            if k[0] == tid
        )

        out.write("\n" + "#" * 100 + "\n")
        out.write(f"TARGET {tid} | N PAIRS = {len(keys)}\n")
        out.write("#" * 100 + "\n\n")

        for pair_i, key in enumerate(keys, start=1):

            arms = by_pair[key]

            assert set(arms) == {
                "correction",
                "natural",
            }

            c = arms["correction"]
            n = arms["natural"]

            assert (
                c["continuation_seed"]
                == n["continuation_seed"]
            )

            out.write("=" * 100 + "\n")
            out.write(
                f"{tid} PAIR {pair_i}/{len(keys)} "
                f"| source_seed={key[1]} "
                f"| continuation_seed={c['continuation_seed']}\n"
            )
            out.write("=" * 100 + "\n\n")

            out.write("ORIGINAL CORRECTION S:\n")
            out.write(c["intervention_text"])
            out.write("\n\n")

            out.write("NATURAL ALTERNATIVE S':\n")
            out.write(n["intervention_text"])
            out.write("\n\n")

            out.write(
                "----- CORRECTION ARM: "
                "BEGINNING OF CONTINUATION -----\n"
            )
            out.write(
                beginning(c["raw_continuation"])
            )
            out.write("\n\n")

            out.write(
                "----- NATURAL ARM: "
                "BEGINNING OF CONTINUATION -----\n"
            )
            out.write(
                beginning(n["raw_continuation"])
            )
            out.write("\n\n")

            out.write(
                "----- CORRECTION ARM: "
                "END OF CONTINUATION -----\n"
            )
            out.write(
                ending(c["raw_continuation"])
            )
            out.write("\n\n")

            out.write(
                "----- NATURAL ARM: "
                "END OF CONTINUATION -----\n"
            )
            out.write(
                ending(n["raw_continuation"])
            )
            out.write("\n\n")


print("WROTE:", OUT)
print("Pairs:", len(by_pair))
print("Records:", len(rows))
