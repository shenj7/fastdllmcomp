#!/usr/bin/env bash
# Fast-dLLM GSM8K evaluation — three configurations for Table 1.
# Run from this repo's root. FAST_DLLM_PATH must point to Fast-dLLM/v1/.
#
# Setup:
#   git clone https://github.com/NVlabs/Fast-dLLM.git /path/to/Fast-dLLM
#   pip install transformers==4.49.0 accelerate==0.34.2 datasets
#   export FAST_DLLM_PATH=/path/to/Fast-dLLM/v1

set -e
export FAST_DLLM_PATH="${FAST_DLLM_PATH:?Set FAST_DLLM_PATH to Fast-dLLM/v1/ directory}"

MODEL="GSAI-ML/LLaDA-8B-Base"
N=1000
SEED=1234
GEN=256

# ── Config A: Cache only (steps=256, block_length=32) ─────────────────────
# Same NFE as baseline (~256), but per-step cost drops via prefix KV cache.
# Compare against: Base (fixed, 256 steps) row in Table 1.
python run_fast_dllm_gsm8k.py \
    --mode cache \
    --steps 256 \
    --block_length 32 \
    --gen_length $GEN \
    --model $MODEL \
    --n_examples $N \
    --seed $SEED \
    --output results_fast_dllm_cache_256steps.txt

# ── Config B: Cache + Parallel, default threshold=0.9 ─────────────────────
# Fast-dLLM's primary result: steps budget=8 total (1 per block of 32 tokens).
# Actual NFE ≈ 8–16 (threshold gates unmasking; at least 1 token unmasked/step).
# Compare against: ARFI rows (~62–173 steps) in Table 1.
python run_fast_dllm_gsm8k.py \
    --mode cache_parallel \
    --steps 8 \
    --block_length 32 \
    --threshold 0.9 \
    --gen_length $GEN \
    --model $MODEL \
    --n_examples $N \
    --seed $SEED \
    --output results_fast_dllm_cache_parallel_8steps.txt

# ── Config C: Cache + Parallel, steps=32 (softer reduction) ───────────────
# Intermediate: 4 steps per block, threshold gating reduces actual NFE.
# Use this if Config B accuracy is too low; comparable to ~62-step rows.
python run_fast_dllm_gsm8k.py \
    --mode cache_parallel \
    --steps 32 \
    --block_length 32 \
    --threshold 0.9 \
    --gen_length $GEN \
    --model $MODEL \
    --n_examples $N \
    --seed $SEED \
    --output results_fast_dllm_cache_parallel_32steps.txt

echo ""
echo "All configs done. Check results_fast_dllm_*.txt for Table 1 rows."
