"""
ARFI + FastDLLM combined evaluation on GSM8K for Table 1.

Combines:
  - FastDLLM prefix KV cache (reduces per-step compute cost)
  - ARFI confidence-conditioned scheduler (reduces step count)

The two methods target orthogonal bottlenecks:
  FastDLLM: per-step cost   (KV cache reuse across denoising steps)
  ARFI:     step count      (adaptive token commit count per step)

Generation loop: generate_with_prefix_cache_arfi()
  - Block-wise semi-autoregressive like FastDLLM (block_length=32)
  - Base commit = 1 token/step per block (same base as standalone ARFI)
  - ARFI multiplier overrides commit count each step from logits
  - threshold parameter dropped; ARFI replaces it as the adaptive mechanism

ARFI rule (applied per step, over currently masked positions):
  mean >= 0.20 AND q25 >= 0.060  ->  commit 4x base
  mean >= 0.12 AND q25 >= 0.045  ->  commit 2x base
  otherwise                      ->  commit 1x base

Same 1000-example subset (seed=1234), 4-shot prompt, gen_length=256
as all other Table 1 rows.

Usage:
    python run_arfi_fastdllm_gsm8k.py \
        --fast_dllm_path /scratch/gilbreth/shen574/Fast-dLLM/v1 \
        --cache_dir /scratch/gilbreth/shen574 \
        --output results_arfi_fastdllm.txt
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
parser.add_argument("--fast_dllm_path", default=None,
                    help="Path to Fast-dLLM v1/ directory (or set FAST_DLLM_PATH env var)")
parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
parser.add_argument("--gen_length", type=int, default=256)
parser.add_argument("--block_length", type=int, default=32,
                    help="Tokens per block. Base commit = block_length tokens / block_length steps = 1/step.")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--remasking", default="low_confidence")
parser.add_argument("--n_examples", type=int, default=1000)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--cache_dir", default=None,
                    help="Root dir for HF model/dataset downloads (e.g. /scratch/gilbreth/shen574)")
parser.add_argument("--output", default="results_arfi_fastdllm.txt")
args = parser.parse_args()

# Redirect HuggingFace downloads before any HF import
if args.cache_dir:
    hf_home = str(Path(args.cache_dir) / "hf_cache")
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_home) / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")
    os.makedirs(hf_home, exist_ok=True)

# ---------------------------------------------------------------------------
# Locate Fast-dLLM
# ---------------------------------------------------------------------------
fast_dllm_path = args.fast_dllm_path or os.environ.get("FAST_DLLM_PATH")
if fast_dllm_path is None:
    print("ERROR: set --fast_dllm_path or FAST_DLLM_PATH to the Fast-dLLM v1/ directory")
    sys.exit(1)

# Import FastDLLM helpers (get_num_transfer_tokens, get_transfer_index)
sys.path.insert(0, str(Path(fast_dllm_path) / "llada"))
try:
    from generate import get_num_transfer_tokens, get_transfer_index
except ImportError as e:
    print(f"ERROR: could not import from Fast-dLLM generate.py: {e}")
    sys.exit(1)

# Import FastDLLM's patched LLaDA model (supports use_cache=True)
sys.path.insert(0, str(Path(fast_dllm_path) / "llada"))
from model.modeling_llada import LLaDAModelLM
from model.configuration_llada import LLaDAConfig

# ---------------------------------------------------------------------------
# ARFI scheduler
# ---------------------------------------------------------------------------
def arfi_commit_count(logits: torch.Tensor,
                      mask_index: torch.Tensor,
                      base_n: torch.Tensor) -> torch.Tensor:
    """
    Compute ARFI-adjusted token commit count for one denoising step.

    Args:
        logits:     (B, L, V) — raw model logits for the slice being decoded
        mask_index: (B, L) bool — True at currently masked positions
        base_n:     (B,) int — base transfer count for this step

    Returns:
        (B,) int — clamped ARFI commit count
    """
    probs = torch.softmax(logits.float(), dim=-1)       # (B, L, V)
    max_probs = probs.max(dim=-1).values                # (B, L)

    # Confidences only at masked positions
    masked_probs = max_probs[mask_index]                # (N_masked,) flattened across batch

    if masked_probs.numel() == 0:
        return base_n

    mean_conf = masked_probs.mean().item()
    q25_conf  = torch.quantile(masked_probs, 0.25).item()

    if mean_conf >= 0.20 and q25_conf >= 0.060:
        mult = 4
    elif mean_conf >= 0.12 and q25_conf >= 0.045:
        mult = 2
    else:
        mult = 1

    remaining = mask_index.sum(dim=1)                   # (B,) — don't commit more than remain
    return torch.clamp(base_n * mult, min=1, max=remaining)


# ---------------------------------------------------------------------------
# Combined generation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_prefix_cache_arfi(
    model,
    prompt: torch.Tensor,
    gen_length: int = 256,
    block_length: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> tuple:
    """
    FastDLLM prefix-cache loop with ARFI adaptive commit count.

    Injection point: immediately after logits are computed each step,
    before get_transfer_index selects which tokens to commit.

    Step budget: block_length steps per block (base = 1 token/step),
    same as standalone ARFI over the full 256-token sequence.
    ARFI multiplier reduces actual step count by committing 2x or 4x per step.

    Returns: (x, nfe)  — full sequence tensor and forward-pass count.
    """
    B, L_prompt = prompt.shape
    x = torch.full((B, L_prompt + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :L_prompt] = prompt.clone()

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    # Base: 1 token per step (block_length steps to unmask block_length tokens)
    steps_per_block = block_length

    nfe = 0

    for block_idx in range(num_blocks):
        current_block_start = L_prompt + block_idx * block_length
        current_block_end   = current_block_start + block_length

        # Pre-compute base token-transfer schedule for this block
        block_mask_index   = (x[:, current_block_start:current_block_end] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block) # (B, steps_per_block)

        # ── First pass: full sequence → build KV cache ───────────────────────
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        nfe += 1

        # Mask index for full sequence, zeroed out beyond current block
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0

        # ARFI on first pass
        arfi_n = arfi_commit_count(output.logits, mask_index, num_transfer_tokens[:, 0])
        x0, transfer_index = get_transfer_index(
            output.logits, temperature, remasking, mask_index, x, arfi_n, None)
        x[transfer_index] = x0[transfer_index]

        # Trim KV cache to prompt + already-completed blocks (current_block_start)
        trimmed_kv = []
        for layer_kv in past_key_values:
            trimmed_kv.append(tuple(kv[:, :, :current_block_start] for kv in layer_kv))
        past_key_values = trimmed_kv

        # ── Cached steps within block ─────────────────────────────────────────
        step_i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break  # block fully unmasked

            nfe += 1

            # Only attend to current block's masked positions
            mask_index = (x[:, current_block_start:] == mask_id)   # relative to current_block_start
            mask_index[:, block_length:] = 0                        # cap at current block end

            # Cheap forward pass reusing prompt KV cache
            logits = model(
                x[:, current_block_start:],
                past_key_values=past_key_values,
                use_cache=True,
            ).logits                                                 # (B, seq_from_block_start, V)

            # ── ARFI injection ────────────────────────────────────────────────
            base_n = num_transfer_tokens[:, min(step_i, steps_per_block - 1)]
            arfi_n = arfi_commit_count(logits, mask_index, base_n)
            # ─────────────────────────────────────────────────────────────────

            x0, transfer_index = get_transfer_index(
                logits, temperature, remasking, mask_index,
                x[:, current_block_start:], arfi_n, None)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]

            step_i += 1

    return x, nfe


# ---------------------------------------------------------------------------
# 4-shot prompt  (identical to run_gsm8k_paper_eval.py)
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
# Load model (FastDLLM's patched LLaDAModelLM — supports use_cache=True)
# ---------------------------------------------------------------------------
print(f"Loading model {args.model} ...")
from transformers import AutoConfig

config = LLaDAConfig.from_pretrained(
    args.model, trust_remote_code=True, cache_dir=args.cache_dir)
config.flash_attention = True

tokenizer = AutoTokenizer.from_pretrained(
    args.model, trust_remote_code=True, cache_dir=args.cache_dir)
model = LLaDAModelLM.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    config=config,
    cache_dir=args.cache_dir,
).to("cuda").eval()

MASK_ID = tokenizer.mask_token_id or 126336

# ---------------------------------------------------------------------------
# Load dataset — identical subset to all other Table 1 rows
# ---------------------------------------------------------------------------
print("Loading GSM8K test split ...")
dataset = load_dataset(
    "gsm8k", "main", split="test",
    cache_dir=str(Path(args.cache_dir) / "hf_cache" / "datasets") if args.cache_dir else None)
all_examples = list(dataset)

random.seed(args.seed)
selected = random.sample(all_examples, args.n_examples)

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
correct     = 0
step_counts = []
times       = []

print(f"Running {args.n_examples} examples | ARFI + FastDLLM prefix cache")
print(f"  gen_length={args.gen_length} | block_length={args.block_length} "
      f"| steps_per_block={args.block_length} (base=1 token/step)")
print(f"  ARFI thresholds: 4x if mean>=0.20,q25>=0.060 | 2x if mean>=0.12,q25>=0.045")
print("-" * 60)

for i, ex in enumerate(selected):
    question = ex["question"]
    gold     = normalize(extract_answer(ex["answer"].split("####")[-1].strip()))

    input_ids = tokenizer(build_prompt(question), return_tensors="pt").input_ids.to("cuda")

    t0 = time.perf_counter()
    out_ids, nfe = generate_with_prefix_cache_arfi(
        model, input_ids,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        remasking=args.remasking,
        mask_id=MASK_ID,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen_ids   = out_ids[0, input_ids.shape[1]:]
    generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
    pred      = normalize(extract_answer(generated))

    if pred == gold:
        correct += 1

    step_counts.append(nfe)
    times.append(elapsed)

    if (i + 1) % 50 == 0:
        print(f"[{i+1:4d}/{args.n_examples}] acc={correct/(i+1)*100:.1f}% | "
              f"last_nfe={nfe} | last_sec={elapsed:.2f}s")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
total_sec  = sum(times)
em         = correct / args.n_examples * 100
mean_steps = sum(step_counts) / len(step_counts)
med_steps  = median(step_counts)
mean_sec   = total_sec / args.n_examples
med_sec    = median(times)

print("\n" + "=" * 60)
print("RESULTS — ARFI + FastDLLM")
print("=" * 60)
print(f"EM accuracy:     {em:.2f}%")
print(f"Mean steps(NFE): {mean_steps:.2f}")
print(f"Median steps:    {med_steps:.2f}")
print(f"Mean sec/ex:     {mean_sec:.2f}")
print(f"Median sec/ex:   {med_sec:.2f}")
print(f"Total seconds:   {total_sec:.1f}")

print("\n--- Table 1 row ---")
print(f"ARFI + FastDLLM | {em:.2f} | {mean_steps:.2f} | {med_steps:.2f} | "
      f"{mean_sec:.2f} | {med_sec:.2f} | {total_sec:.1f}")

with open(args.output, "w") as f:
    f.write(f"method=ARFI+FastDLLM gen_length={args.gen_length} block_length={args.block_length} "
            f"seed={args.seed} n={args.n_examples}\n")
    f.write(f"em={em:.4f} mean_steps={mean_steps:.4f} median_steps={med_steps:.4f} "
            f"mean_sec={mean_sec:.4f} median_sec={med_sec:.4f} total_sec={total_sec:.4f}\n")
print(f"\nResults saved to {args.output}")
