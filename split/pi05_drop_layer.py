"""Remove one recorded layer from π0.5 recordings (after the feature choice), rewriting each file atomically.

    python split/pi05_drop_layer.py --layer final $VLA_DATA/pi05/record/*/
"""

import argparse
import os
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", required=True)
    p.add_argument("dirs", nargs="+")
    args = p.parse_args()
    for d in args.dirs:
        files = sorted(Path(d).glob("task*_ep*.pt"))
        for f in files:
            e = torch.load(f, weights_only=False)
            if args.layer not in e:
                continue
            del e[args.layer]
            tmp = f.with_suffix(".tmp")
            torch.save(e, tmp)
            os.replace(tmp, f)
        print(f"{d}: {len(files)} files", flush=True)


if __name__ == "__main__":
    main()
