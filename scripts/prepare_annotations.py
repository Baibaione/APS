#!/usr/bin/env python3
"""Validate/install canonical benchmark annotations, without rewriting questions.

Accepts JSON or JSONL with the fields documented in docs/data.md. For arbitrary
new data use --allow-new-protocol; reference selected frames then cannot be reused.
"""
import argparse
import json
from pathlib import Path
import shutil
from aps.infer import DATASETS
from aps.io import read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=list(DATASETS), required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-new-protocol', action='store_true')
    a = p.parse_args()
    raw = a.input.read_bytes()
    docs = [json.loads(x) for x in raw.splitlines() if x.strip()] if a.input.suffix == '.jsonl' else json.loads(raw)
    cfg = DATASETS[a.dataset]
    if len(docs) != cfg['n_questions'] or len({d[cfg['id_key']] for d in docs}) != len(docs):
        raise ValueError('Incorrect benchmark question count or duplicate IDs')
    for doc in docs:
        for key in ['question', cfg['video_key'], *cfg['group_fields']]:
            if key not in doc:
                raise ValueError(f'Missing field: {key}')
        if not cfg['check_answer'](doc):
            raise ValueError('Invalid gold answer')
    repo = Path(__file__).resolve().parents[1]
    expected = read(repo / 'assets/input_manifest.json')[a.dataset]
    if not a.allow_new_protocol:
        if sha(a.input) != expected['annotation_sha256']:
            raise ValueError('Annotation differs from the reference snapshot. Use matching reference inputs, '
                             'or --allow-new-protocol and regenerate scores/selections. Do not mix protocols.')
    if a.output.exists():
        if read(a.output) != docs:
            raise ValueError('Existing annotation differs; choose a new output')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.input.suffix == '.jsonl':
        write(a.output, docs)
    elif a.input.resolve() != a.output.resolve():
        shutil.copyfile(a.input, a.output)
    print('Validated', len(docs), 'questions; SHA256', sha(a.output))


if __name__ == '__main__':
    main()
