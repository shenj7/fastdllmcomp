"""
Fast-dLLM baseline evaluation on GSM8K for Table 1.

Matches the 1000-example subset (random.seed(1234)) and 4-shot prompt
from run_gsm8k_paper_eval.py exactly. Swaps only the generation backend.

Usage:
    # Cache only (steps unchanged, per-step cost drops):
    python run_fast_dllm_gsm8k.py --mode cache --steps 256 --block_length 32

    # Cache + parallel decoding (steps AND per-step cost drop):
    python run_fast_dllm_gsm8k.py --mode cache_parallel --steps 8 --block_length 32 --threshold 0.9

Prerequisites:
    git clone https://github.com/NVlabs/Fast-dLLM.git
    export FAST_DLLM_PATH=/path/to/Fast-dLLM/v1
"""

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path
from statistics import median

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir", default=None,
                    help="Root directory for all downloads (HF models, datasets). "
                         "Overrides HF_HOME. E.g. /scratch/gilbreth/shen574")
parser.add_argument("--fast_dllm_path", default=None,
                    help="Path to Fast-dLLM v1/ directory (or set FAST_DLLM_PATH env var)")
parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
parser.add_argument("--mode", choices=["baseline", "cache", "cache_parallel"],
                    default="cache_parallel",
                    help="baseline=no cache; cache=prefix cache only; cache_parallel=cache+parallel decoding")
parser.add_argument("--steps", type=int, default=8,
                    help="Total denoising steps budget (actual NFE may be lower with parallel)")
parser.add_argument("--gen_length", type=int, default=256)
parser.add_argument("--block_length", type=int, default=32,
                    help="Must divide gen_length. KV cache reuse spans within each block. "
                         "32 (8 blocks) gives meaningful inter-block reuse; 256 = single block.")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--threshold", type=float, default=0.9,
                    help="Confidence threshold for parallel unmasking (mode=cache_parallel)")
parser.add_argument("--remasking", default="low_confidence")
parser.add_argument("--n_examples", type=int, default=1000)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--output", default="fast_dllm_results.txt")
args = parser.parse_args()

# Redirect all HuggingFace downloads before any HF library is imported.
if args.cache_dir:
    hf_home = str(Path(args.cache_dir) / "hf_cache")
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_home) / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")
    os.makedirs(hf_home, exist_ok=True)

# ---------------------------------------------------------------------------
# Locate Fast-dLLM generate.py
# ---------------------------------------------------------------------------
import os

fast_dllm_path = args.fast_dllm_path or os.environ.get("FAST_DLLM_PATH")
if fast_dllm_path is None:
    print("ERROR: set --fast_dllm_path or FAST_DLLM_PATH to the Fast-dLLM v1/ directory")
    sys.exit(1)

sys.path.insert(0, str(Path(fast_dllm_path) / "llada"))

try:
    from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
except ImportError as e:
    print(f"ERROR: could not import Fast-dLLM generate.py from {fast_dllm_path}/llada: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4-shot prompt (LLaDA recipe — identical to run_gsm8k_paper_eval.py)
# ---------------------------------------------------------------------------
FOUR_SHOT_PREFIX = """\
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. The answer is 72.

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. The answer is 10.

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: In the beginning, Betty has only 100 / 2 = $50. Betty's grandparents gave her 15 * 2 = $30. This means, Betty needs 100 - 50 - 15 - 30 = $5 more. The answer is 5.

Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read tomorrow?
Answer: Altogether, Julie read 12 + 24 = 36 pages. There are 120 - 36 = 84 pages left. She should read 84 / 2 = 42 pages tomorrow. The answer is 42.

Question: {question}
Answer:"""


def build_prompt(question: str) -> str:
    return FOUR_SHOT_PREFIX.format(question=question)


def extract_answer(text: str) -> str:
    """Extract final numeric answer from 'The answer is X.' or last number."""
    m = re.search(r"[Tt]he answer is\s*([\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def normalize(ans: str) -> str:
    try:
        return str(int(float(ans)))
    except Exception:
        return ans.strip()


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading model {args.model} ...")
from transformers import AutoModel

tokenizer = AutoTokenizer.from_pretrained(
    args.model, trust_remote_code=True, cache_dir=args.cache_dir)
model = AutoModel.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    cache_dir=args.cache_dir,
).to("cuda").eval()

MASK_ID = tokenizer.mask_token_id
if MASK_ID is None:
    # LLaDA uses token 126336
    MASK_ID = 126336

# ---------------------------------------------------------------------------
# Load dataset — identical subset to run_gsm8k_paper_eval.py
# ---------------------------------------------------------------------------
print("Loading GSM8K test split ...")
dataset = load_dataset("gsm8k", "main", split="test",
                       cache_dir=str(Path(args.cache_dir) / "hf_cache" / "datasets") if args.cache_dir else None)
all_examples = list(dataset)

random.seed(args.seed)
selected = random.sample(all_examples, args.n_examples)

# ---------------------------------------------------------------------------
# Choose generation function
# ---------------------------------------------------------------------------
def run_generation(input_ids):
    """
    Returns (output_ids, nfe) where nfe = actual number of forward passes.
    All Fast-dLLM generate functions return nfe as second element.
    """
    kwargs = dict(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        remasking=args.remasking,
        mask_id=MASK_ID,
    )
    if args.mode == "baseline":
        out, nfe = generate(model, input_ids, **kwargs)
    elif args.mode == "cache":
        out, nfe = generate_with_prefix_cache(model, input_ids, **kwargs)
    else:  # cache_parallel
        kwargs["threshold"] = args.threshold
        out, nfe = generate_with_dual_cache(model, input_ids, **kwargs)
    return out, nfe


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
correct = 0
step_counts = []
times = []

print(f"Running {args.n_examples} examples | mode={args.mode} | steps={args.steps} | "
      f"block_length={args.block_length} | threshold={getattr(args, 'threshold', 'N/A')}")
print("-" * 60)

for i, ex in enumerate(selected):
    question = ex["question"]
    gold_raw = ex["answer"]
    gold = normalize(extract_answer(gold_raw.split("####")[-1].strip()))

    prompt = build_prompt(question)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids, nfe = run_generation(input_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Decode only the generated portion
    gen_ids = out_ids[0, input_ids.shape[1]:]
    generated = tokenizer.decode(gen_ids, skip_special_tokens=True)

    pred = normalize(extract_answer(generated))
    is_correct = (pred == gold)
    if is_correct:
        correct += 1

    step_counts.append(nfe)
    times.append(elapsed)

    if (i + 1) % 50 == 0:
        running_acc = correct / (i + 1) * 100
        print(f"[{i+1:4d}/{args.n_examples}] acc={running_acc:.1f}% | "
              f"last_nfe={nfe} | last_sec={elapsed:.2f}s")

# ---------------------------------------------------------------------------
# Aggregate metrics (matching run_gsm8k_paper_eval.py format)
# ---------------------------------------------------------------------------
total_sec = sum(times)
em = correct / args.n_examples * 100
mean_steps = sum(step_counts) / len(step_counts)
med_steps = median(step_counts)
mean_sec = total_sec / args.n_examples
med_sec = median(times)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Method:          Fast-dLLM ({args.mode})")
print(f"Model:           {args.model}")
print(f"Steps budget:    {args.steps} | block_length={args.block_length}")
print(f"Threshold:       {getattr(args, 'threshold', 'N/A')}")
print(f"N examples:      {args.n_examples} (seed={args.seed})")
print(f"EM accuracy:     {em:.2f}%")
print(f"Mean steps(NFE): {mean_steps:.2f}")
print(f"Median steps:    {med_steps:.2f}")
print(f"Mean sec/ex:     {mean_sec:.2f}")
print(f"Median sec/ex:   {med_sec:.2f}")
print(f"Total seconds:   {total_sec:.1f}")

# Table 1 row (copy-paste ready)
print("\n--- Table 1 row ---")
print(f"Fast-dLLM ({args.mode}) | {em:.2f} | {mean_steps:.2f} | {med_steps:.2f} | "
      f"{mean_sec:.2f} | {med_sec:.2f} | {total_sec:.1f}")

# Save to file
with open(args.output, "w") as f:
    f.write(f"mode={args.mode} steps={args.steps} block_length={args.block_length} "
            f"threshold={getattr(args,'threshold','N/A')} seed={args.seed}\n")
    f.write(f"em={em:.4f} mean_steps={mean_steps:.4f} median_steps={med_steps:.4f} "
            f"mean_sec={mean_sec:.4f} median_sec={med_sec:.4f} total_sec={total_sec:.4f}\n")
print(f"\nResults saved to {args.output}")
