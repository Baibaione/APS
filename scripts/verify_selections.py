#!/usr/bin/env python3
"""Recompute every released APS selection using the original BLIP score caches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from aps import APSConfig, select_frames
from aps.io import read, sha, write


def check(item):
    ds, i, score, frames, expected = item
    for budget in [16,32]:
        actual = select_frames(score,frames,APSConfig(budget=budget,beta=.6 if ds=='longvideobench' else .4))
        if actual != expected[budget]:
            return {'dataset':ds,'index':i,'budget':budget}
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root', type=Path, default=Path('cache'))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--report', type=Path, default=Path('outputs/selection_verification.json'))
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=read(root/'assets/input_manifest.json')
    items=[]
    for ds,meta in manifest.items():
        files={x:a.cache_root/ds/'blip'/f'{x}.json' for x in ['scores','frames']}
        for key,path in files.items():
            if sha(path)!=meta[key+'_sha256']:
                raise ValueError(f'Historical cache mismatch: {ds}/{key}')
        scores,frames=read(files['scores']),read(files['frames'])
        expected={k:read(root/f'assets/selections/{ds}/aps_k{k}.json') for k in [16,32]}
        if len(scores)!=len(frames) or len(scores)!=meta['questions']:
            raise ValueError('Invalid cache coverage')
        items += [(ds,i,s,f,{k:expected[k][i] for k in [16,32]}) for i,(s,f) in enumerate(zip(scores,frames))]
    with ProcessPoolExecutor(a.workers) as pool:
        differences=[x for x in pool.map(check,items,chunksize=8) if x]
    report={'questions':len(items),'selection_comparisons':len(items)*2,'mismatches':differences}
    write(a.report,report)
    print(report)
    if differences:
        raise SystemExit(1)


if __name__=='__main__':
    main()
