# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def main(args):
    agg = defaultdict(list)
    for seed_dir in args.seed_dirs:
        path = os.path.join(seed_dir, "average_scores.json")
        with open(path) as f:
            data = json.load(f)
        print(f"\n--- {seed_dir} ---")
        for name in sorted(data):
            value = data[name]
            print(f"  {name:<20}: {value:.6f}")
            agg[name].append(value)

    print("\n--- Averages across seeds ---")
    for name in sorted(agg):
        print(f"  {name:<20}: {float(np.mean(agg[name])):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Average each metric across per-seed average_scores.json files."
    )
    parser.add_argument(
        "--seed_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One directory per seed, each containing average_scores.json.",
    )
    main(parser.parse_args())
