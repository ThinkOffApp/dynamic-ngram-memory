#!/usr/bin/env python3
"""Materialize a provenance-checked adapter namespace without editing the pinned export.

This first step contains no v4 semantics and is not wired into the production ABI.
Later adapter patches must be recorded explicitly, not applied to SOURCE.json files.
"""
import argparse
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKER = ".notes-adapter-generated.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def derive(output, verify=False):
    source = json.loads((ROOT / "SOURCE.json").read_text())
    blobs, origins = {}, {}
    for entry in source["files"]:
        path = pathlib.PurePosixPath(entry["path"])
        if path.parts[:2] != ("zig", "src") or ".." in path.parts:
            raise ValueError("invalid pinned path")
        original = ROOT / "module" / path
        if original.is_symlink():
            raise ValueError("symlink in pinned export")
        data = original.read_bytes()
        if digest(data) != entry["sha256"]:
            raise ValueError("pinned export hash mismatch: " + str(path))
        relative = str(pathlib.PurePosixPath(*path.parts[2:]))
        blobs[relative], origins[relative] = data, entry["path"]
    for relative in ("server/model_text.zig", "standalone_erase.zig", "standalone/erase_places.zig"):
        local = ROOT / "module/zig/src" / relative
        if local.is_symlink():
            raise ValueError("symlink in local glue")
        blobs[relative], origins[relative] = local.read_bytes(), str(local.relative_to(ROOT))
    receipt = dict(format="notes-adapter-derivation/1", source_commit=source["commit"],
                   semantic_patches=[], files=[dict(path=p, origin=origins[p], sha256=digest(blobs[p]))
                                               for p in sorted(blobs)])
    output = pathlib.Path(output).absolute()
    if output.is_symlink() or any(p.is_symlink() for p in output.parents):
        raise ValueError("symlink in output path")
    marker = output / MARKER
    if verify:
        if json.loads(marker.read_text()) != receipt:
            raise ValueError("derivation receipt mismatch")
    elif output.exists() and any(output.iterdir()):
        if not marker.exists() or json.loads(marker.read_text()).get("format") != receipt["format"]:
            raise ValueError("refusing to overwrite a non-generated directory")
    for relative, data in blobs.items():
        target = output / relative
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise ValueError("symlink in generated target")
        if verify:
            if target.read_bytes() != data:
                raise ValueError("generated source mismatch: " + relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if not verify:
        marker.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{'Verified' if verify else 'Derived'} {len(blobs)} files; no semantic patches; source {source['commit']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "module/zig/src/adapter")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    derive(args.out, args.verify)
