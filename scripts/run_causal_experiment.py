import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "models/Qwen3.5-35B-A3B"

MANIFEST = ROOT / "results/causal_experiment_frozen.json"

SOURCE = (
    ROOT
    / "repos/value_leakage_data/static/data/giraffes/qwen3.5-35"
    / "v1_giraffes_accurate.json"
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--targets",
    nargs="+",
    required=True,
)

parser.add_argument(
    "--output",
    required=True,
)

parser.add_argument(
    "--dry-run",
    action="store_true",
)

args = parser.parse_args()


OUT = ROOT / args.output


with MANIFEST.open(encoding="utf-8") as f:
    manifest = json.load(f)

with SOURCE.open(encoding="utf-8") as f:
    source = json.load(f)


targets = {
    t["blind_id"]: t
    for t in manifest["targets"]
}


for tid in args.targets:
    if tid not in targets:
        raise SystemExit(f"Unknown/non-frozen target: {tid}")


MAX_NEW_TOKENS = int(
    manifest["sampling"]["max_new_tokens"]
)

assert MAX_NEW_TOKENS == 8192


# ------------------------------------------------------------
# Existing-output handling.
#
# This makes the experiment safely resumable. A completed
# (target, source seed, arm) is never generated twice.
# ------------------------------------------------------------

completed = set()

if OUT.exists():
    print(
        f"Existing output found: {OUT}\n"
        "Resuming from completed generations.",
        flush=True,
    )

    with OUT.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                r = json.loads(line)
            except Exception as e:
                raise RuntimeError(
                    f"Invalid JSON at {OUT}:{lineno}"
                ) from e

            key = (
                r["blind_id"],
                int(r["source_regeneration_seed"]),
                r["arm"],
            )

            if key in completed:
                raise RuntimeError(
                    f"Duplicate completed record already in output: {key}"
                )

            completed.add(key)


# ------------------------------------------------------------
# Construct expected experimental jobs.
# ------------------------------------------------------------

jobs = []

for tid in args.targets:
    t = targets[tid]

    for pair in t["pairs"]:

        source_seed = int(
            pair["source_regeneration_seed"]
        )

        continuation_seed = int(
            pair["continuation_seed"]
        )

        # Deterministic order chosen only from the already-frozen
        # source seed. This does not use any causal outcome.
        #
        # We reset RNG independently for each arm, so both members
        # of a pair start from the SAME continuation seed.
        if source_seed % 2 == 0:
            arms = ["correction", "natural"]
        else:
            arms = ["natural", "correction"]

        for arm in arms:
            jobs.append({
                "blind_id": tid,
                "target": t,
                "pair": pair,
                "arm": arm,
                "source_seed": source_seed,
                "continuation_seed":
                    continuation_seed,
            })


print("\nEXPERIMENT CHECK")
print("----------------")

for tid in args.targets:
    t = targets[tid]
    print(
        tid,
        "|", t["direction"],
        "| pairs =", t["n_pairs"],
        "| generations =", 2 * t["n_pairs"],
    )

print("\nJobs requested:", len(jobs))
print(
    "Already completed:",
    sum(
        (
            j["blind_id"],
            j["source_seed"],
            j["arm"],
        ) in completed
        for j in jobs
    ),
)
print(
    "Remaining:",
    sum(
        (
            j["blind_id"],
            j["source_seed"],
            j["arm"],
        ) not in completed
        for j in jobs
    ),
)


if args.dry_run:
    print("\nDRY RUN ONLY — no model loaded, no generation.")
    raise SystemExit(0)


# ------------------------------------------------------------
# Load model.
# ------------------------------------------------------------

print("\nLoading tokenizer...", flush=True)

tok = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

print("Loading model...", flush=True)

model = Qwen3_5MoeForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

model.eval()

input_device = model.get_input_embeddings().weight.device

print(
    "Model loaded. Input device:",
    input_device,
    flush=True,
)


# Normalize EOS IDs.
eos_ids = model.generation_config.eos_token_id

if isinstance(eos_ids, int):
    eos_ids = [eos_ids]

eos_ids = set(eos_ids or [])


# ------------------------------------------------------------
# Cache each target's exact P and chat prefix.
# ------------------------------------------------------------

base_prefixes = {}

for tid in args.targets:
    t = targets[tid]

    prompt = source["prompts"][t["direction"]]

    generation_prefix = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert generation_prefix.endswith(
        "<|im_start|>assistant\n<think>\n"
    )

    base_prefixes[tid] = (
        generation_prefix
        + t["prefix_reasoning"]
    )


# ------------------------------------------------------------
# Generation.
# ------------------------------------------------------------

worker_start = time.time()
newly_completed = 0


for job_idx, job in enumerate(jobs, start=1):

    tid = job["blind_id"]
    t = job["target"]
    pair = job["pair"]
    arm = job["arm"]

    source_seed = job["source_seed"]
    continuation_seed = job["continuation_seed"]

    key = (
        tid,
        source_seed,
        arm,
    )

    if key in completed:
        continue


    if arm == "correction":
        intervention = t["original_correction_S"]

    elif arm == "natural":
        intervention = pair[
            "natural_alternative_S_prime"
        ]

    else:
        raise RuntimeError(arm)


    exact_text = (
        base_prefixes[tid]
        + intervention
    )


    enc = tok(
        exact_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    enc = {
        k: v.to(input_device)
        for k, v in enc.items()
    }


    # Matched/common random seed.
    random.seed(continuation_seed)
    torch.manual_seed(continuation_seed)
    torch.cuda.manual_seed_all(
        continuation_seed
    )


    print(
        "\n"
        f"START "
        f"{tid} "
        f"source_seed={source_seed} "
        f"continuation_seed={continuation_seed} "
        f"arm={arm} "
        f"[job {job_idx}/{len(jobs)}]",
        flush=True,
    )


    start = time.time()

    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=
                model.generation_config.pad_token_id,
            eos_token_id=
                model.generation_config.eos_token_id,
            use_cache=True,
        )


    elapsed = time.time() - start

    input_len = enc["input_ids"].shape[1]

    new_ids = out[0, input_len:]

    n_tokens = int(new_ids.shape[0])

    raw = tok.decode(
        new_ids,
        skip_special_tokens=False,
    )


    last_token = (
        int(new_ids[-1].item())
        if n_tokens
        else None
    )

    hit_eos = (
        last_token in eos_ids
        if last_token is not None
        else False
    )

    hit_max_tokens = (
        n_tokens >= MAX_NEW_TOKENS
        and not hit_eos
    )


    record = {
        "blind_id": tid,
        "source_row_index":
            t["source_row_index"],
        "source_line_index":
            t["source_line_index"],
        "direction":
            t["direction"],

        "source_regeneration_seed":
            source_seed,

        "continuation_seed":
            continuation_seed,

        "arm":
            arm,

        "intervention_text":
            intervention,

        "n_input_tokens":
            int(input_len),

        "n_generated_tokens":
            n_tokens,

        "hit_eos":
            hit_eos,

        "hit_max_tokens":
            hit_max_tokens,

        "elapsed_seconds":
            elapsed,

        "tokens_per_second":
            (
                n_tokens / elapsed
                if elapsed > 0
                else None
            ),

        "raw_continuation":
            raw,
    }


    # One complete JSON record is appended only after
    # generation finishes.
    with OUT.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


    completed.add(key)
    newly_completed += 1


    preview = (
        raw
        .replace("\n", " ")
        .replace("\r", " ")
    )[:180]


    print(
        f"DONE  {tid} "
        f"source_seed={source_seed} "
        f"arm={arm} "
        f"| tokens={n_tokens} "
        f"| sec={elapsed:.1f} "
        f"| tok/s={n_tokens/elapsed:.2f} "
        f"| EOS={hit_eos} "
        f"| MAX={hit_max_tokens}",
        flush=True,
    )

    print(
        "preview:",
        preview,
        flush=True,
    )


total_elapsed = time.time() - worker_start

print("\n" + "=" * 90)
print("WORKER FINISHED")
print("New generations:", newly_completed)
print("Total elapsed sec:", round(total_elapsed, 1))
print("Output:", OUT)
print("=" * 90)
