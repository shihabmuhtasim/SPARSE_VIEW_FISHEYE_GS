#!/usr/bin/env python3
"""Download and verify the exact Flowers COLMAP inputs, without downloading RGB."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import tempfile
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads((Path(__file__).resolve().parent.parent / "provenance/flowers_geometry_asset.json").read_text())
    target = args.scene_dir.expanduser().resolve() / "sparse/0"
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in pin["files_sha256"].items():
        existing = target / name
        if existing.exists() and hashlib.sha256(existing.read_bytes()).hexdigest() != expected:
            raise ValueError("Existing different geometry will not be overwritten: " + str(existing))
    with tempfile.TemporaryDirectory(prefix="flowers-geometry-") as directory:
        archive = Path(directory) / "geometry.tar.gz"
        urllib.request.urlretrieve(pin["asset_url"], archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != pin["sha256"]:
            raise ValueError("Geometry archive checksum mismatch")
        with tarfile.open(archive) as tar:
            if set(tar.getnames()) != set(pin["files_sha256"]):
                raise ValueError("Unexpected files in geometry archive")
            # Write only whitelisted basenames; never extract arbitrary archive paths.
            for name, expected in pin["files_sha256"].items():
                member = tar.getmember(name)
                if Path(name).name != name or not member.isfile():
                    raise ValueError("Invalid geometry archive member")
                data = tar.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("Geometry member checksum mismatch: " + name)
                (target / name).write_bytes(data)
    print("Verified exact Flowers geometry in", target)
    print("Place the original 149 RGB photographs under", args.scene_dir / "images")


if __name__ == "__main__":
    main()
