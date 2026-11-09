import json
from pathlib import Path
import pytest
from aps.infer import DATASETS, author_functions, load_results, summarize
from aps.io import manifest, write
from aps.experiments import commands


def test_truncated_checkpoint_is_repaired_only_explicitly(tmp_path):
    p = tmp_path/'rank0.jsonl'
    good = json.dumps({'status':'ok','index':0,'method':'aps'})+'\n'
    p.write_text(good+'{"unfinished":')
    with pytest.raises(ValueError):
        load_results(tmp_path)
    assert len(load_results(tmp_path, repair_tail=True)) == 1
    assert p.read_text() == good


def test_immutable_resume_manifest(tmp_path):
    path = tmp_path/'manifest.json'
    manifest(path, {'alpha':1})
    manifest(path, {'alpha':1})
    with pytest.raises(ValueError):
        manifest(path, {'alpha':1.5})


def test_mlvu_macro_differs_from_micro_and_rejects_incomplete(tmp_path):
    counts = [200,206,352,355,259,539,263]
    docs, rows = [], []
    for task,n in enumerate(counts):
        for _ in range(n):
            i = len(docs)
            docs.append({'question_id':f'Q{i}', 'answer':'A'})
            pred = 'A' if task == 0 else 'B'
            rows.append(dict(status='ok', index=i, id=f'Q{i}', method='aps', answer='A',
                             prediction=pred, correct=pred=='A', task_type=f'task{task}',
                             unparsed_fallback=False, uniform_frame_fallback=False))
    write(tmp_path/'questions.json',docs)
    write(tmp_path/'manifest.json',dict(dataset='mlvu',annotation=str(tmp_path/'questions.json'),
          indices=list(range(2174)),methods=['aps'],group_fields=['task_type']))
    result_path = tmp_path/'rank0.jsonl'
    result_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result = summarize(tmp_path)['methods']['aps']
    assert result['main_table_accuracy_percent'] == pytest.approx(100/7)
    assert result['accuracy_percent'] == pytest.approx(100*200/2174)
    result_path.write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
    with pytest.raises(ValueError, match='未完成'):
        summarize(tmp_path)


def test_prompt_and_parser_contract():
    base = Path(__file__).resolve().parents[1]/'src/aps/eval_utils'
    mlvu = author_functions(base/'mlvu.py', DATASETS['mlvu']['function_names'])
    question = 'Choose?\n(A) One\n(B) Two\n'
    assert mlvu['mlvu_doc_to_text']({'question':question},DATASETS['mlvu']['prompt_kwargs']) == question+'\nOnly give the best option.\nBest option: ('
    assert mlvu['extract_characters_regex']('Reading a book.') == ''
    assert mlvu['extract_characters_regex']('(B)') == 'B'
    lvb = author_functions(base/'longvideobench.py', DATASETS['longvideobench']['function_names'])
    assert lvb['longvideobench_doc_to_text']({'question':'Q?','candidates':['one','two']},DATASETS['longvideobench']['prompt_kwargs']) == "Q?\nA. one\nB. two\nAnswer with the option's letter from the given choices directly.\n"


def test_main_matrix_and_gpu_commands(tmp_path):
    root = Path(__file__).resolve().parents[1]
    jobs = json.loads((root/'configs/main_table.json').read_text())['jobs']
    assert len(jobs) == 9
    assert {j['nframes'] for j in jobs if j['model']=='minicpm'} == {16}
    for job in jobs:
        assert job['beta'] == (.6 if job['dataset']=='longvideobench' else .4)
        cmds = commands(job,tmp_path,'0;1',8)
        assert cmds[1][cmds[1].index('--gpus')+1] == '0;1'
        assert '--annotation' in cmds[0]


def test_ablation_resource_settings_match_historical_manifests():
    root = Path(__file__).resolve().parents[1]
    jobs = json.loads((root/'configs/ablations.json').read_text())['jobs']
    k64 = [j for j in jobs if j['axis']=='frames' and j['nframes']==64]
    large = [j for j in jobs if j['axis']=='scale' and j.get('model_size')=='32B']
    assert len(k64) == len(large) == 2
    assert all(j['max_model_len']==49152 and j['kv_cache_gib']==4 for j in k64)
    assert all(j['tensor_parallel_size']==2 and j['kv_cache_gib']==6 for j in large)
