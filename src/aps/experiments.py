"""Print or execute portable experiment commands. Default is plan-only."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from .io import read


def identity(job):
    return {k: v for k, v in job.items() if k not in ('id', 'axis', 'source_manifest_sha256', 'model_size')}


def commands(job, root, gpus, limit, methods=None):
    j = dict(job)
    if methods:
        j['methods'] = methods.split(',')
    signature = hashlib.sha256(json.dumps(identity(j), sort_keys=True).encode()).hexdigest()[:12]
    ds = j['dataset']
    cache = root / 'cache' / ds / j['scorer']
    output = root / 'outputs' / f'{j["model"]}_{ds}_{signature}' / (f'smoke{limit}' if limit else 'full')
    selection_key = hashlib.sha256(json.dumps({k: j[k] for k in ['dataset', 'scorer', 'nframes', 'alpha', 'beta', 'penalty', 'variant']}, sort_keys=True).encode()).hexdigest()[:12]
    selection = root / 'cache' / 'selected' / f'{ds}_{selection_key}.json'
    select = [sys.executable, '-m', 'aps.select', '--scores', str(cache/'scores.json'), '--frames', str(cache/'frames.json'),
              '--annotation', str(root/f'data/{ds}/questions.json'), '--output', str(selection),
              '--budget', str(j['nframes']), '--alpha', str(j['alpha']), '--beta', str(j['beta']),
              '--penalty', str(j['penalty']), '--variant', j['variant'], '--workers', '8']
    run = [sys.executable, '-m', 'aps.infer', 'run', '--dataset', ds, '--data-root', str(root/'data'),
           '--model', str(root/j['model_path']), '--candidates', str(cache/'frames.json'), '--selected', str(selection),
           '--methods', ','.join(j['methods']), '--gpus', gpus, '--tensor-parallel-size', str(j['tensor_parallel_size']),
           '--nframes', str(j['nframes']), '--max-model-len', str(j['max_model_len']), '--kv-cache-gib', str(j['kv_cache_gib']),
           '--seed', str(j['seed']), '--limit', str(limit), '--run-dir', str(output)]
    return [select, run] if 'aps' in j['methods'] else [run]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/main_table.json')
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--job', help='Exact job ID')
    p.add_argument('--axis', help='Select one ablation axis')
    p.add_argument('--gpus', default='0', help='Quote semicolon-separated data-parallel groups')
    p.add_argument('--methods', choices=['aps', 'uniform', 'aps,uniform'])
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    jobs = [j for j in read(a.config)['jobs'] if (not a.job or j['id'] == a.job) and (not a.axis or j.get('axis') == a.axis)]
    if not jobs:
        p.error('No matching experiments')
    seen = set()
    for j in jobs:
        if any(len(g.split(',')) != j['tensor_parallel_size'] for g in a.gpus.split(';')):
            p.error(f'{j["id"]} needs GPU groups of size {j["tensor_parallel_size"]}; select jobs with matching TP')
        cmds = commands(j, a.root.resolve(), a.gpus, a.limit, a.methods)
        key = tuple(cmds[-1])
        if key in seen:
            continue
        seen.add(key)
        print('#', j['id'], flush=True)
        for cmd in cmds:
            print(shlex.join(cmd), flush=True)
            if a.execute:
                subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
