#!/usr/bin/env python3
"""Freeze another scene's repository split without inspecting test quality."""
from pathlib import Path
import argparse
import json
from prepare_study import poses, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="Lowercase scene name, for example chairs or globe")
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--import-baseline", type=Path, help="Verify and preserve a teammate's already frozen split")
    args = parser.parse_args()
    if not args.scene.replace("_", "").isalnum():
        parser.error("Use a simple scene name containing letters, digits or underscores")
    geometry = args.scene_dir / "sparse/0"
    names = sorted(poses(geometry / "images.bin"))
    tests = names[::8]
    train = [name for i, name in enumerate(names) if i % 8]
    hashes = {name: sha(geometry / name) for name in ["cameras.bin", "images.bin", "points3D.bin"]}
    if args.import_baseline:
        old = json.loads(args.import_baseline.read_text())
        if old["train_images"] != train or sorted(old["test_images"]) != tests:
            raise ValueError("Existing split differs from the fixed every-eighth repository split; coordinate this before running")
        for name, value in hashes.items():
            if old["reconstruction"]["model_files_sha256"][name] != value:
                raise ValueError("Teammate COLMAP hash differs: " + name)
    if len(train) < 96:
        raise ValueError("This four-budget study needs at least 96 eligible training images")
    for name in names:
        if Path(name).name != name or not (args.scene_dir / "images" / name).is_file():
            raise ValueError("Expected original flat image filename: " + name)
    manifest = {"scene": args.scene, "manifest_version": 1, "train_images": train, "test_images": tests,
                "reconstruction": {"model_files_sha256": hashes, "image_ordering": "lexicographic original filename"},
                "geometry_limitation": "Frozen full-scene COLMAP includes test-assisted geometry; test observations and training gradients are excluded."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {args.scene}: {len(names)} total, {len(train)} training candidates, {len(tests)} tests")
    print(args.output)


if __name__ == "__main__":
    main()
