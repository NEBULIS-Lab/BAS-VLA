#!/usr/bin/env python3
"""Inspect intervention recipes, export prompt variants, or transform an RGB image."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from bas_vla.integrations.openpi_libero import apply_image_shift, load_distractor_assets
from bas_vla.preserving.appearance import APPEARANCE_SHIFT_PRESETS, get_appearance_shift_spec
from bas_vla.runtime.scene import SceneVariation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    image = commands.add_parser("image", help="apply an image-space operator to an already oriented RGB frame")
    image.add_argument("--input", type=Path, required=True)
    image.add_argument("--output", type=Path, required=True)
    image.add_argument("--preset", choices=sorted(APPEARANCE_SHIFT_PRESETS), required=True)
    image.add_argument("--seed", type=int, default=7)
    image.add_argument("--distractor-manifest", type=Path)
    prompt = commands.add_parser("prompt", help="export an exact catalog instruction")
    prompt.add_argument("--catalog", type=Path, default=REPO_ROOT / "configs/interventions/semantic_pairs.json")
    prompt.add_argument("--pair-id", required=True)
    prompt.add_argument("--condition", choices=("clean", "control", "break", "paraphrase", "redundant", "format"), required=True)
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps({"appearance": APPEARANCE_SHIFT_PRESETS,
            "geometry": {mode: vars(SceneVariation(mode=mode)) for mode in ("layout_shift", "view_jitter")}}, indent=2))
    elif args.command == "prompt":
        payload = json.loads(args.catalog.read_text())
        matches = [entry for entry in payload["pairs"] if entry["pair_id"] == args.pair_id]
        if len(matches) != 1:
            raise ValueError("pair-id must select exactly one catalog entry")
        print(matches[0][args.condition + "_instruction"])
    else:
        spec = get_appearance_shift_spec(args.preset)
        if spec["scene_properties"] is not None:
            raise ValueError("preset includes simulator scene properties; apply it through the appearance evaluator")
        with Image.open(args.input) as image:
            array = np.asarray(image.convert("RGB"))
        result = apply_image_shift(array, spec["image_shift"], args.seed, load_distractor_assets(args.distractor_manifest))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            Image.fromarray(result).save(handle, format="PNG")
        print(json.dumps({"preset": args.preset, "seed": args.seed, "image_shift": spec["image_shift"]}, indent=2))


if __name__ == "__main__":
    main()
