"""Construct the peripheral distractor bank from LIBERO demonstration frames."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image


def build_distractor_bank(specification, dataset_root, output_dir):
    """Read the specified RGB frames without rotation and export exact crops.

    Source paths are relative to dataset_root, and exported manifest paths are
    relative to output_dir. Existing outputs are never overwritten.
    """
    import h5py

    root = Path(dataset_root).resolve()
    output = Path(output_dir)
    entries = specification["assets"]
    if not entries or len({entry["name"] for entry in entries}) != len(entries):
        raise ValueError("asset specification needs unique nonempty entries")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    prepared = []
    for entry in entries:
        name = entry["name"]
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("asset names must be simple filenames")
        relative = Path(entry["source_hdf5"])
        source = (root / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(root):
            raise ValueError("source_hdf5 must be relative to dataset_root")
        target = output / f"{name}.png"
        if target.exists():
            raise FileExistsError(target)
        frame_idx = entry["frame_idx"]
        with h5py.File(source, "r") as handle:
            frames = handle["data"][entry["demo_key"]]["obs"]["agentview_rgb"]
            if type(frame_idx) is not int or not 0 <= frame_idx < len(frames):
                raise ValueError("frame_idx is out of range")
            image = Image.fromarray(frames[frame_idx])
        box = entry["crop_box_xyxy"]
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("crop_box_xyxy requires four integer coordinates")
        left, top, right, bottom = box
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError("crop box lies outside the source frame")
        crop = image.crop(tuple(box))
        prepared.append((target, crop, {**entry, "asset_path": target.name, "asset_size": list(crop.size)}))
    output.mkdir(parents=True, exist_ok=True)
    for target, crop, _ in prepared:
        with target.open("xb") as handle:
            crop.save(handle, format="PNG")
    manifest = {"assets": [entry for _, _, entry in prepared]}
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest
