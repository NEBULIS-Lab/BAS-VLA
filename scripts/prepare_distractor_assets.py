#!/usr/bin/env python3
"""Export the documented peripheral LIBERO crops using a local dataset root."""
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from bas_vla.preserving.assets import build_distractor_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=REPO_ROOT / "configs/interventions/distractor_crops.json")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_distractor_bank(json.loads(args.spec.read_text()), args.dataset_root, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
