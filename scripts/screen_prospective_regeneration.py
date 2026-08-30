import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/Qwen3.5-35B-A3B"

MANIFEST = ROOT / "results/prospective_active_decision_targets_frozen.json"

SOURCE = (
    ROOT
    / "repos/value_leakage_data/static/data/giraffes/qwen3.5-35"
    / "v1_giraffes_accurate.json"
)

SEED_BASE = {
    "B008": 9000,
    "B011": 9100,
    "B012": 9200,
    "B028": 9300,
    "B035": 9400,
    "B036": 9500,
    "B039": 9600,
    "B047": 9700,
}

parser = argparse.ArgumentParser()
parser.add_argument("--targets", nargs="+", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--n", type=int, default=10)
args = parser.parse_args()

OUT = ROOT / args.output

if OUT.exists():
    raise SystemExit(f"{OUT} already exists; refusing to overwrite.")

with MANIFEST.open(encoding="utf-8") as f:
    manifest = json.load(f)

with SOURCE.open(encoding="utf-8") as f:
    source = json.load(f)

targets = {t["blind_id"]: t for t in manifest["targets"]}

print("Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

print("Loading model...", flush=True)
model = Qwen3_5MoeForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.eval()


def first_nonempty_line(text):
    for line in text.splitlines():
        if line.strip():
            return line.rstrip() + "\n"
    return ""


def make_inputs(t):
    prompt = source["prompts"][t["direction"]]

    generation_prefix = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert generation_prefix.endswith(
        "<|im_start|>assistant\n<think>\n"
    )

    # Exact reasoning prefix P, immediately before original correction S.
    text = generation_prefix + t["prefix_reasoning"]

    enc = tok(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in enc.items()}


for tid in args.targets:
    t = targets[tid]
    inputs = make_inputs(t)

    print("\n" + "=" * 100)
    print(tid, "|", t["direction"])
    print("ORIGINAL S:", t["correction_episode"].strip())
    print("=" * 100, flush=True)

    for i in range(args.n):
        seed = SEED_BASE[tid] + i

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=160,
                pad_token_id=model.generation_config.pad_token_id,
                eos_token_id=model.generation_config.eos_token_id,
            )

        new_ids = out[0, inputs["input_ids"].shape[1]:]

        raw = tok.decode(
            new_ids,
            skip_special_tokens=False,
        )

        candidate = first_nonempty_line(raw)

        print(
            f"[{tid} {i+1:02d}/{args.n}] "
            f"seed={seed}: {candidate.strip()}",
            flush=True,
        )

        record = {
            "blind_id": tid,
            "source_row_index": t["source_row_index"],
            "direction": t["direction"],
            "seed": seed,
            "original_correction_S": t["correction_episode"],
            "candidate_S_prime": candidate,
            "raw_generation": raw,
        }

        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

print("\nDONE:", OUT, flush=True)
