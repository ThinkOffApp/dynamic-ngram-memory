#!/usr/bin/env python3
"""Stage the pinned TensorFold host and verify this module's exact source export."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=pathlib.Path)
    parser.add_argument('--source-repository', type=pathlib.Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--zig', default='zig')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'SOURCE.json').read_text())
    for entry in manifest['files']:
        rel = pathlib.PurePosixPath(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe module path')
        data = (ROOT / 'module' / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Module checksum mismatch: ' + str(rel))
    dest = args.destination.resolve()
    if dest.exists():
        raise ValueError('Destination exists; choose a new directory: ' + str(dest))
    if args.check:
        version = subprocess.check_output([args.zig, 'version'], text=True).strip()
        if version != '0.17.0':
            raise ValueError('Expected Zig 0.17.0, found ' + version)
    source = str(args.source_repository.resolve()) if args.source_repository else manifest['repository']
    dest.mkdir(parents=True)
    run(['git', 'init', '--quiet', str(dest)])
    run(['git', '-C', str(dest), 'fetch', '--quiet', '--depth=1', source, manifest['commit']])
    run(['git', '-C', str(dest), 'checkout', '--quiet', '--detach', 'FETCH_HEAD'])
    for entry in manifest['files']:
        rel = entry['path']
        data = (dest / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Pinned host differs from module export: ' + rel)
    print(f"Verified {len(manifest['files'])} module files at {manifest['commit']}", flush=True)
    if args.check:
        run([args.zig, 'build', 'test-server-cpu', '--summary', 'all'], cwd=dest)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
