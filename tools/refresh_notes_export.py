#!/usr/bin/env python3
"""Mechanically refresh the pristine Notes export; adapters must live outside these files."""
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    source = pathlib.Path(sys.argv[1]).resolve()
    ref = sys.argv[2]
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", ref + "^{commit}"], text=True).strip()
    manifest_path = ROOT / "SOURCE.json"
    manifest = json.loads(manifest_path.read_text())
    previous = {entry["path"] for entry in manifest["files"]}
    # Existing host test/entry exports remain pinned too; add every new Notes component.
    all_paths = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", "-r", "--name-only", revision], text=True).splitlines()
    selected = sorted(p for p in all_paths if p in previous or p.startswith("zig/src/server/notes/")
                      or p in ("zig/src/server/notes_records.zig", "zig/src/server/notes_erase_places.zig",
                               "zig/src/server/json.zig", "zig/src/server/json_parse.zig",
                               "zig/src/server/json_write.zig", "zig/src/server/clock.zig"))
    if not selected or not all(p.startswith("zig/") and ".." not in pathlib.PurePosixPath(p).parts
                               for p in selected):
        raise ValueError("invalid export paths")
    # Resolve all blobs before changing the checkout. Never follow export-file symlinks.
    blobs = {p: subprocess.check_output(["git", "-C", str(source), "show", revision + ":" + p])
             for p in selected}
    for p in previous | set(selected):
        target = ROOT / "module" / p
        if target.is_symlink():
            raise ValueError("symlink in export: " + p)
    for p in sorted(previous - set(selected)):
        (ROOT / "module" / p).unlink()
    for p, data in blobs.items():
        target = ROOT / "module" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest["commit"] = revision
    manifest["files"] = [{"path": p, "sha256": hashlib.sha256(blobs[p]).hexdigest()} for p in selected]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(selected)} pristine source files at {revision}")


if __name__ == "__main__":
    main()
