"""Solve for the largest output_len (capped) that fits in KV-cache VRAM for a given batch_size.

Uses the same accounting as vram_estimation.py. Prints the chosen output_len to stdout
so it can be consumed by shell scripts:

    out=$(python benchmarks/max_output_len.py --model gpt-oss-20b --batch-size 256)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_specs import PRESET_MODELS
from vram_estimation import calculate_kv_cache_size, calculate_model_size


# Map local-checkout names to PRESET_MODELS keys.
MODEL_NAME_ALIASES = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "Qwen3-Coder-30B-A3B-Instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit": "cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
}


def resolve_model(name: str):
    key = MODEL_NAME_ALIASES.get(name, name)
    if key not in PRESET_MODELS:
        sys.exit(f"[max_output_len] model '{name}' (resolved to '{key}') not in PRESET_MODELS")
    return PRESET_MODELS[key]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Local model name or PRESET_MODELS key.")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--input-len", type=int, default=1)
    p.add_argument("--max-output-len", type=int, default=16384)
    p.add_argument("--total-vram-gb", type=float, default=24.0,
                   help="Per-GPU VRAM (RTX 4090 = 24).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                   help="Matches vLLM --gpu-memory-utilization default.")
    p.add_argument("--activation-overhead-gb", type=float, default=2.0,
                   help="Reserved for CUDA context, activations, and scheduler overhead.")
    args = p.parse_args()

    model = resolve_model(args.model)
    model_size_gb = calculate_model_size(model)
    available_gb = args.total_vram_gb * args.gpu_memory_utilization
    kv_budget_gb = available_gb - model_size_gb - args.activation_overhead_gb

    print(f"[max_output_len] model={model.name} weights={model_size_gb:.2f} GB "
          f"available={available_gb:.2f} GB kv_budget={kv_budget_gb:.2f} GB",
          file=sys.stderr)

    if kv_budget_gb <= 0:
        sys.exit(f"[max_output_len] no KV budget left after weights+overhead "
                 f"({model_size_gb:.2f} + {args.activation_overhead_gb:.2f} > {available_gb:.2f} GB)")

    def fits(output_len: int) -> bool:
        ctx = args.input_len + output_len
        kv_gb = calculate_kv_cache_size(model, args.batch_size, ctx)
        return kv_gb <= kv_budget_gb

    if not fits(1):
        sys.exit(f"[max_output_len] batch_size={args.batch_size} cannot fit even output_len=1")

    if fits(args.max_output_len):
        print(args.max_output_len)
        return

    lo, hi = 1, args.max_output_len
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    print(lo)


if __name__ == "__main__":
    main()
