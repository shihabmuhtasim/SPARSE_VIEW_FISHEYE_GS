#!/usr/bin/env python3
"""Fetch the pinned training engine and unchanged partner PCA selector."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(destination, *args):
    subprocess.run(["git", "-C", str(destination), *args], check=True)


def partner():
    pin = json.loads((PACKAGE / "provenance/partner_script.json").read_text())
    target = HERE / "partner_reference" / pin["path"]
    if not target.exists() or sha(target) != pin["sha256"]:
        url = (pin["repository"].replace("https://github.com/", "https://raw.githubusercontent.com/")
               + "/" + pin["commit"] + "/" + pin["path"])
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != pin["sha256"]:
            raise ValueError("Pinned partner selector checksum mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    print("Verified unchanged partner PCA selector:", pin["sha256"])


def engine(destination):
    pin = json.loads((PACKAGE / "provenance/upstream_provenance.json").read_text())
    expected = json.loads((PACKAGE / "provenance/training_source_sha256.json").read_text())
    if not destination.exists():
        destination.mkdir(parents=True)
        git(destination, "init", "-q")
        git(destination, "remote", "add", "origin", pin["repository"] + ".git")
        # The Windows viewer and teaser images are unnecessary for headless training.
        git(destination, "sparse-checkout", "init", "--cone")
        git(destination, "sparse-checkout", "set", "arguments", "scene", "utils",
            "gaussian_renderer", "lpipsPyTorch", "submodules")
        git(destination, "fetch", "--filter=blob:none", "--depth=1", "origin", pin["commit"])
        git(destination, "checkout", "--detach", "-q", "FETCH_HEAD")
    head = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if head != pin["commit"]:
        raise ValueError("Destination is not the pinned checkout; use a new empty destination")
    if not all((destination / name).is_file() and sha(destination / name) == h
               for name, h in expected.items()):
        patch = PACKAGE / "provenance/training_from_upstream.patch"
        if sha(patch) != pin["sha256"]:
            raise ValueError("Training patch checksum mismatch")
        git(destination, "apply", "--check", str(patch))
        git(destination, "apply", str(patch))
    for name, h in expected.items():
        if sha(destination / name) != h:
            raise ValueError("Patched training file differs from the executed study: " + name)
    print("Verified", len(expected), "training files at", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--partner-only", action="store_true")
    args = parser.parse_args()
    if not args.partner_only and args.destination is None:
        parser.error("Supply --destination or --partner-only")
    partner()
    if not args.partner_only:
        engine(args.destination.expanduser().resolve())


if __name__ == "__main__":
    main()
