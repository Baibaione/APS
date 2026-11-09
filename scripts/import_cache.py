#!/usr/bin/env python3
"""Import checksum-verified annotations and relevance caches.

Videos are linked when available, never copied. The default source layout is
data/<dataset>/questions.json and cache/<dataset>/blip/{scores,frames}.json.
"""
import argparse
import shutil
from pathlib import Path
from aps.io import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--layout', choices=['aps', 'aks'], default='aps')
    a = p.parse_args()
    expected = read(Path(__file__).resolve().parents[1] / 'assets/input_manifest.json')
    copies, video_links = [], []
    for ds, identity in expected.items():
        source_data = a.source / ('datasets' if a.layout == 'aks' else 'data') / ds
        source_cache = a.source / ('outscores' if a.layout == 'aks' else 'cache') / ds / 'blip'
        annotation = source_data / ('include_frame_idx.json' if a.layout == 'aks' else 'questions.json')
        inputs = [(annotation, a.root / f'data/{ds}/questions.json', 'annotation_sha256')]
        inputs += [(source_cache / f'{name}.json', a.root / f'cache/{ds}/blip/{name}.json', name + '_sha256') for name in ['scores', 'frames']]
        for src, dest, key in inputs:
            if sha(src) != identity[key]:
                raise ValueError(f'Reference input checksum mismatch: {src}')
            if dest.exists() and sha(dest) != identity[key]:
                raise ValueError(f'Refusing to replace different data: {dest}')
            copies.append((src, dest))
        src = source_data / ('data' if a.layout == 'aks' and ds == 'videomme' else 'videos')
        dest = a.root / f'data/{ds}/videos'
        video_links.append((src, dest))
    # Validate the complete bundle before copying any files.
    for src, dest in copies:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
    for src, dest in video_links:
        if src.is_dir() and not dest.exists() and not dest.is_symlink():
            dest.symlink_to(src.resolve(), target_is_directory=True)
        print(dest.parent.name, 'annotations and BLIP caches verified; video directory:', dest)


if __name__ == '__main__':
    main()
