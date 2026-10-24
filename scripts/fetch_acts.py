"""Pull activation .npz files off the Modal volume.

    python scripts/fetch_acts.py --run-id main

Uses the Modal Python SDK rather than `modal volume get`, which on this setup
creates the destination directory and then aborts because the destination
directory exists -- so it cannot succeed regardless of --force.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

VOLUME = "swayambhu-cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--volume", default=VOLUME)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    remote = f"acts/{args.run_id}"
    out = Path(args.out or f"data/activations_{args.run_id}")
    out.mkdir(parents=True, exist_ok=True)

    vol = modal.Volume.from_name(args.volume)
    entries = [e for e in vol.listdir(remote) if e.path.endswith(".npz")]
    entries.sort(key=lambda e: e.path)
    print(f"{len(entries)} files under {remote} -> {out}")

    done = skipped = 0
    total = 0
    for i, e in enumerate(entries, 1):
        dest = out / Path(e.path).name
        if dest.exists() and not args.overwrite and dest.stat().st_size > 0:
            skipped += 1
            continue
        tmp = dest.with_suffix(".part")
        n = 0
        with open(tmp, "wb") as fh:
            for chunk in vol.read_file(e.path):
                fh.write(chunk)
                n += len(chunk)
        tmp.replace(dest)
        done += 1
        total += n
        if i % 25 == 0 or i == len(entries):
            print(f"  {i}/{len(entries)}  ({total/1e6:.0f} MB)")

    print(f"done: {done} downloaded, {skipped} already present, {total/1e6:.0f} MB")


if __name__ == "__main__":
    main()
