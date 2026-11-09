#!/usr/bin/env python3
"""Convert the seven official MLVU MCQ JSON files to the documented schema.

Match the recorded question/video order by hashes, without bundling question text.
After conversion regenerate scores: serialization/source revisions can differ
from the historical annotation snapshot.
"""
import argparse
import hashlib
from pathlib import Path
from aps.io import read, write

TASKS = ['plotQA','needle','ego','count','order','anomaly_reco','topic_reasoning']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--json-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('data/mlvu/questions.json'))
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    index=read(repo/'assets/question_manifest/mlvu.json')
    source={}
    for number,task in enumerate(TASKS,1):
        for row in read(a.json_dir/f'{number}_{task}.json'):
            candidates=row['candidates']
            video=row.get('video',row.get('video_name'))
            if video is None:
                raise ValueError('Missing video name in source annotation')
            video=Path(video).name
            question=row['question']+'\n'+''.join(f'({chr(65+i)}) {s}\n' for i,s in enumerate(candidates))
            answer=row['answer']
            if answer in candidates:
                answer=chr(65+candidates.index(answer))
            elif answer not in ['A','B','C','D']:
                raise ValueError('Gold answer does not match the choices')
            rel=f'{number}_{task}/{video}'
            key=(rel,hashlib.sha256(question.encode()).hexdigest())
            if key in source:
                raise ValueError('Duplicate source question/video identity')
            source[key]=dict(task_type=task,video_name=video,video_relpath=rel,
                             duration=float(row['duration']),question=question,candidates=candidates,answer=answer)
    docs=[]
    for record in index:
        key=(record['video'],record['question_sha256'])
        if key not in source:
            raise ValueError(f'Source version mismatch at {record["id"]}; do not silently reorder or replace questions')
        docs.append(dict(question_id=record['id'],**source.pop(key)))
    if source or len(docs)!=2174:
        raise ValueError('MLVU source coverage mismatch')
    if a.output.exists() and read(a.output)!=docs:
        raise ValueError('Output already contains different annotations')
    write(a.output,docs)
    print('Converted 2174 questions. Regenerate caches for this annotation snapshot.')


if __name__=='__main__':
    main()
