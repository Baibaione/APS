#!/usr/bin/env python3
"""Download local model snapshots from ModelScope by default."""
import argparse
from pathlib import Path

MODELS = {
    'qwen7b': ('Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen2.5-VL-7B-Instruct'),
    'qwen3b': ('Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen2.5-VL-3B-Instruct'),
    'qwen32b': ('Qwen/Qwen2.5-VL-32B-Instruct', 'Qwen2.5-VL-32B-Instruct'),
    'internvl': ('OpenGVLab/InternVL3-8B', 'InternVL3-8B'),
    'minicpm': ('OpenBMB/MiniCPM-V-4', 'MiniCPM-V-4'),
    'blip': ('Salesforce/blip-itm-large-coco', 'blip-itm-large-coco'),
    'clip': ('openai/clip-vit-base-patch32', 'clip-vit-base-patch32'),
    'blip2': ('Salesforce/blip2-itm-vit-g', 'blip2-itm-vit-g'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', choices=list(MODELS), default=['qwen7b', 'internvl', 'minicpm'])
    p.add_argument('--provider', choices=['modelscope', 'huggingface'], default='modelscope')
    p.add_argument('--output-dir', type=Path, default=Path('models'))
    p.add_argument('--revision', default=None, help='Optional explicit snapshot revision')
    a = p.parse_args()
    if a.provider == 'modelscope':
        from modelscope import snapshot_download
    else:
        from huggingface_hub import snapshot_download
    for name in a.models:
        repo, folder = MODELS[name]
        # Some scorer mirrors are not available on ModelScope; use the documented HF path explicitly.
        kw = dict(local_dir=str(a.output_dir / folder))
        if a.revision:
            kw['revision'] = a.revision
        snapshot_download(repo, **kw)
        print(name, a.output_dir / folder)


if __name__ == '__main__':
    main()
