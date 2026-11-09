#!/usr/bin/env python3
"""Import an existing experiment tree into the portable data/cache layout.

Copies only annotations, score/frame JSONs. Videos are linked, never copied.
Run from the repository root; source must point to the former AKS directory.
"""
import argparse
import shutil
from pathlib import Path
from aps.io import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('.'))
    a = p.parse_args()
    expected = read(a.root / 'assets/input_manifest.json')
    for ds, identity in expected.items():
        inputs = [(a.source / f'datasets/{ds}/include_frame_idx.json', a.root / f'data/{ds}/questions.json', 'annotation_sha256')]
        inputs += [(a.source / f'outscores/{ds}/blip/{name}.json', a.root / f'cache/{ds}/blip/{name}.json', name + '_sha256') for name in ['scores', 'frames']]
        for src, dest, key in inputs:
            if sha(src) != identity[key]:
                raise ValueError(f'Historical input checksum mismatch: {src}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and sha(dest) != identity[key]:
                raise ValueError(f'Refusing to replace different data: {dest}')
            shutil.copy2(src, dest)
        # Link only when real videos exist (an input-only archive may omit them).
        src = a.source / f'datasets/{ds}' / ('data' if ds == 'videomme' else 'videos')
        dest = a.root / f'data/{ds}/videos'
        if src.is_dir() and not dest.exists() and not dest.is_symlink():
            dest.symlink_to(src.resolve(), target_is_directory=True)
        print(ds, 'annotations and BLIP caches verified; video directory:', dest)


if __name__ == '__main__':
    main()
