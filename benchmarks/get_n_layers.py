"""Print n_layers for a preset model.

Used by bench scripts to feed the pipeline's --num-layers arg without
having to hard-code per-model values.

Usage:
    n=$(python3 benchmarks/get_n_layers.py gpt-oss-20b)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from max_output_len import resolve_model


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: get_n_layers.py <model-name>")
    print(resolve_model(sys.argv[1]).n_layers)
