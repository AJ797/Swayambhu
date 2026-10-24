"""Gate 1 against the LOCAL GPU backend.

    python scripts/gate1_local.py
    SWAY_LOCAL_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/gate1_local.py

Identical experiment code to the Modal path -- only the runner differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swayambhu.gate1 import run_gate1  # noqa: E402
from swayambhu.local_runner import LocalRunner  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="chunk size; lower it if CUDA runs out of memory")
    ap.add_argument("--model", default=None)
    ap.add_argument("--items", type=int, default=8)
    args = ap.parse_args()

    kw = {"micro_batch": args.micro_batch}
    if args.model:
        kw["model_name"] = args.model
    runner = LocalRunner(**kw)

    result = run_gate1(runner, out_dir=Path("data"), n_items=args.items)
    out = Path("data/gate1_local.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
