"""NVIDIA CUDA / upstream vLLM evaluator for the APS experiment protocol.

Ported from the historical evaluator. GPU execution on NVIDIA is pending
validation; see docs/reproducibility.md. No platform plugins are loaded.
"""
import argparse
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

METHODS = ('aps', 'uniform')
POST_PROMPT = "Answer with the option's letter from the given choices directly.\n"

DATASETS = {
    'longvideobench': {
        'data_dir': 'datasets/longvideobench', 'video_subdir': 'videos',
        'annotation': 'include_frame_idx.json', 'reference': 'lvb_val.json',
        'n_questions': 1337, 'id_key': 'id', 'video_key': 'video_path', 'video_suffix': '',
        'strict_keys': ('id', 'video_path'), 'diff_keys': ('question', 'candidates', 'correct_choice'),
        'function_names': {'longvideobench_doc_to_text', 'get_multi_choice_info', 'parse_multi_choice_response'},
        'doc_to_text': 'longvideobench_doc_to_text',
        'prompt_kwargs': {'pre_prompt': '', 'post_prompt': POST_PROMPT},
        'max_tokens': 32,
        'group_fields': ('duration_group', 'question_category'),
        'float_frames': False, 'random_parser': True,
        'gold_of': lambda doc: chr(65 + doc['correct_choice']),
        'check_answer': lambda doc: 0 <= doc['correct_choice'] < len(doc['candidates']),
    },
    'videomme': {
        'data_dir': 'datasets/videomme', 'video_subdir': 'data',
        'annotation': 'include_frame_idx.json', 'reference': 'videomme.json',
        'n_questions': 2700, 'id_key': 'question_id', 'video_key': 'videoID', 'video_suffix': '.mp4',
        'strict_keys': ('question_id', 'videoID'), 'diff_keys': ('question', 'options', 'answer'),
        'function_names': {'videomme_doc_to_text', 'extract_characters_regex'},
        'doc_to_text': 'videomme_doc_to_text',
        'prompt_kwargs': {'post_prompt': "\nAnswer with the option's letter from the given choices directly."},
        'max_tokens': 16,
        'group_fields': ('duration', 'domain', 'sub_category', 'task_type'),
        'float_frames': True, 'random_parser': False,
        'gold_of': lambda doc: doc['answer'],
        'check_answer': lambda doc: doc['answer'] in tuple('ABCDEFGH'),
    },
    # MLVU dev 集（lmms-eval 协议）：7 个 MCQ 任务共 2174 题，question 内嵌 "(A) 选项"，answer 为字母
    'mlvu': {
        'data_dir': 'datasets/mlvu', 'video_subdir': 'videos',
        'annotation': 'include_frame_idx.json', 'reference': 'mlvu.json',
        'n_questions': 2174, 'id_key': 'question_id', 'video_key': 'video_relpath', 'video_suffix': '',
        'strict_keys': ('question_id', 'video_name'), 'diff_keys': ('question', 'candidates', 'answer'),
        'function_names': {'mlvu_doc_to_text', 'extract_characters_regex'},
        'doc_to_text': 'mlvu_doc_to_text',
        'prompt_kwargs': {'pre_prompt': '', 'post_prompt': '\nOnly give the best option.\nBest option: ('},
        'max_tokens': 16,
        'group_fields': ('task_type',),
        'float_frames': False, 'random_parser': False,
        'gold_of': lambda doc: doc['answer'],
        'check_answer': lambda doc: doc['answer'] in tuple('ABCD'),
    },
}


def as_int_frames(rows):
    """VideoMME 的缓存帧网格/选帧是浮点（如 29.0），统一归一化为 int。"""
    return [[int(round(x)) for x in row] for row in rows]


# 模型类型：与数据集正交，从模型 config.json 的 architectures 自动识别
MODEL_TYPES = {
    'qwen2vl': {'nframes': 32},
    # InternVL 系列：每帧 vllm 强制 resize 到 448×448 单 tile，256 token/帧；
    # uniform 用官方 get_index 居中采样（与 lmms-eval internvl 一致）
    'internvl': {'nframes': 32, 'frame_size': 448, 'tokens_per_frame': 256},
    # MiniCPM-V-4：视频 = 16 张图像列表 + 空格分隔占位符（BPE 合词防护），
    # 每帧动态切 tile，tokens_per_frame 为估算上限
    'minicpmv': {'nframes': 16, 'frame_size': 448, 'tokens_per_frame': 128},
}
ARCH_TO_MODEL_TYPE = {
    'Qwen2VLForConditionalGeneration': 'qwen2vl',
    'Qwen2_5_VLForConditionalGeneration': 'qwen2vl',
    'InternVLChatModel': 'internvl',
    'MiniCPMV': 'minicpmv',
}


def detect_model_type(model_path):
    cfg = json.loads((Path(model_path) / 'config.json').read_text())
    arch = (cfg.get('architectures') or [''])[0]
    if arch not in ARCH_TO_MODEL_TYPE:
        raise ValueError(f'不支持的模型架构: {arch}（支持: {", ".join(ARCH_TO_MODEL_TYPE)}）')
    return ARCH_TO_MODEL_TYPE[arch]


def build_prompt(model_type, processor, text):
    """Preserve the APS experiment prompts for the three answering models.

    Qwen uses its HF chat template; InternVL uses its default system message
    and video placeholder. MiniCPM image placeholders are added by the worker.
    """
    if model_type == 'qwen2vl':
        return processor.apply_chat_template([
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': text}]}],
            tokenize=False, add_generation_prompt=True)
    if model_type == 'internvl':
        return ("<|im_start|>system\n"
                "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>\n"
                "<|im_start|>user\n<video>\n" + text + "<|im_end|>\n"
                "<|im_start|>assistant\n")
    if model_type == 'minicpmv':
        # MiniCPM 视频 = 16 张图像 + 空格分隔占位符（BPE 合词防护）
        # build_prompt 里 nframes 通过 processor 未知，这里由 worker 在调用前处理好占位符；
        # 为简化，直接返回带空格占位符的文本（worker 按 nframes 在 text 前拼好）
        return text  # minicpmv 的占位符由 worker 单独拼
    raise ValueError(f'Unsupported answering model type: {model_type}')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def author_functions(path, names):
    """Load only pure prompt/answer functions, without importing lmms-eval."""
    import numpy as np
    import re
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'作者 utils.py 中缺少函数: {path}')
    module = ast.Module(body=nodes, type_ignores=[])
    env = {'np': np, 'random': random, 're': re}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)
    return env


def native_resize(height, width):
    """Historical Qwen preprocessing with qwen-vl-utils 0.0.8 smart_resize settings.

    max_pixels = original height*width; min_pixels = 128*28*28.
    The later HF processor has no spatial resize left to perform.
    """
    factor, min_pixels, max_pixels = 28, 128 * 28 * 28, height * width
    if height <= 0 or width <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError(f'无效视频尺寸 {height}x{width}')
    rh = max(factor, round(height / factor) * factor)
    rw = max(factor, round(width / factor) * factor)
    if rh * rw > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        rh, rw = math.floor(height / beta / factor) * factor, math.floor(width / beta / factor) * factor
    elif rh * rw < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        rh, rw = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    if min(rh, rw) <= 0:
        raise ValueError(f'预处理尺寸无效: {rh}x{rw}')
    return rh, rw


def assigned(indices, rank, world):
    return [i for i in indices if i % world == rank]


def validate_selected(indices, total_frames):
    if not indices or any(type(i) is not int or i < 0 or i >= total_frames for i in indices):
        raise ValueError(f'APS 帧编号越界或无效; 视频帧数={total_frames}, indices={indices[:8]}')
    if indices != sorted(set(indices)):
        raise ValueError('APS 帧编号必须按时间排序且没有重复')


def check_annotation_identity(docs, reference, dscfg):
    """Use author annotations, verify that downloaded videos/questions align.

    Option wording or answer edits must not silently replace the author's
    evaluation input. Record those differences instead of mixing two versions.
    """
    if len(docs) != len(reference):
        raise ValueError(f'作者标注与下载标注数量不匹配: {len(docs)} vs {len(reference)}')
    differences = []
    for i, (doc, other) in enumerate(zip(docs, reference)):
        for key in dscfg['strict_keys']:
            if doc[key] != other[key]:
                raise ValueError(f'下载数据与作者标注的题目/视频顺序不一致: index={i}, field={key}')
        fields = [key for key in dscfg['diff_keys']
                  if doc[key] != other[key]]
        if fields:
            differences.append({'index': i, 'id': doc[dscfg['id_key']], 'fields': fields})
    return differences


def load_results(run_dir, repair_tail=False, rank=None):
    records = {}
    paths = sorted(Path(run_dir).glob('rank*.jsonl')) if rank is None else [Path(run_dir) / f'rank{rank}.jsonl']
    for path in paths:
        if not path.is_file():
            continue
        data = path.read_bytes()
        if data and not data.endswith(b'\n'):
            if not repair_tail:
                raise ValueError(f'结果文件含中断时的未完成行: {path}')
            data = data[:data.rfind(b'\n') + 1]
            path.write_bytes(data)
        for line in data.splitlines():
            r = json.loads(line)
            if r.get('status') != 'ok':
                continue
            key = (r['index'], r['method'])
            if key in records:
                raise ValueError(f'重复预测: {key}, {path}')
            records[key] = r
    return records


def summarize(run_dir):
    manifest = read_json(Path(run_dir) / 'manifest.json')
    methods = manifest['methods']
    group_fields = manifest.get('group_fields', ['duration_group', 'question_category'])
    expected = {(i, m) for i in manifest['indices'] for m in methods}
    records = load_results(run_dir)
    if set(records) != expected:
        missing, extra = expected - set(records), set(records) - expected
        raise ValueError(f'评测未完成，不能报告完整准确率: 缺少{len(missing)}条，多出{len(extra)}条')
    out = {'questions': len(manifest['indices']), 'methods': {}}
    for method in methods:
        rows = [records[(i, method)] for i in manifest['indices']]
        docs = read_json(manifest['annotation'])
        dscfg = DATASETS[manifest['dataset']]
        for r in rows:
            doc = docs[r['index']]
            gold = dscfg['gold_of'](doc)
            if r['id'] != doc[dscfg['id_key']] or r['answer'] != gold or r['correct'] != (r['prediction'] == gold):
                raise ValueError('Prediction identity / gold / correctness mismatch')
        correct = sum(r['correct'] for r in rows)
        groups = {}
        for field in group_fields:
            for label in sorted({str(r[field]) for r in rows}):
                group = [r for r in rows if str(r[field]) == label]
                groups[f'{field}:{label}'] = {
                    'n': len(group), 'accuracy_percent': 100 * sum(r['correct'] for r in group) / len(group)}
        out['methods'][method] = {'correct': correct, 'n': len(rows),
                                 'accuracy_percent': 100 * correct / len(rows),
                                 'unparsed_fallbacks': sum(r['unparsed_fallback'] for r in rows),
                                 'uniform_frame_fallbacks': sum(r['uniform_frame_fallback'] for r in rows),
                                 'groups': groups}
    for method in methods:
        result = out['methods'][method]
        result['main_table_accuracy_percent'] = result['accuracy_percent']
        if manifest['dataset'] == 'mlvu':
            task = {k:v for k,v in result['groups'].items() if k.startswith('task_type:')}
            if len(manifest['indices']) == DATASETS['mlvu']['n_questions']:
                if len(task) != 7:
                    raise ValueError('MLVU requires seven tasks for full macro accuracy')
                result['task_macro_accuracy_percent'] = sum(v['accuracy_percent'] for v in task.values()) / 7
                result['main_table_accuracy_percent'] = result['task_macro_accuracy_percent']
            else:
                result['main_table_accuracy_percent'] = None
    if 'aps' in methods and 'uniform' in methods:
        out['micro_gain_percentage_points'] = (out['methods']['aps']['accuracy_percent'] -
                                                out['methods']['uniform']['accuracy_percent'])
        a = out['methods']['aps']['main_table_accuracy_percent']
        u = out['methods']['uniform']['main_table_accuracy_percent']
        if a is not None and u is not None:
            out['gain_percentage_points'] = a - u
    write_json(Path(run_dir) / 'summary.json', out)
    for method in methods:
        r = out['methods'][method]
        print(f"{method}: micro {r['correct']}/{r['n']} = {r['accuracy_percent']:.3f}%")
        if 'task_macro_accuracy_percent' in r:
            print(f"{method}: MLVU main-table macro = {r['task_macro_accuracy_percent']:.3f}%")
    if 'gain_percentage_points' in out:
        print(f"APS 增益: {out['gain_percentage_points']:+.3f} 个百分点")
    print('结果:', Path(run_dir) / 'summary.json')
    return out


def choose_paths(args):
    model = Path(args.model).resolve()
    model_type = detect_model_type(model)
    if model_type not in ('qwen2vl', 'internvl', 'minicpmv'):
        raise ValueError('Only Qwen2.5-VL, InternVL3, and MiniCPM-V-4 are supported')
    data = Path(args.data_root).resolve() / args.dataset
    annotation = Path(args.annotation).resolve() if args.annotation else data / 'questions.json'
    reference = Path(args.reference).resolve() if args.reference else annotation
    utils = Path(__file__).resolve().parent / 'eval_utils' / (args.dataset + '.py')
    paths = dict(model=str(model), model_type=model_type, dataset=args.dataset,
                 annotation=str(annotation), reference_annotation=str(reference),
                 video_root=str(Path(args.video_root).resolve() if args.video_root else data / 'videos'),
                 candidate_frames=str(Path(args.candidates).resolve()), utils=str(utils),
                 selected_frames_map={})
    if 'aps' in args.methods.split(','):
        if not args.selected:
            raise ValueError('--selected is required for APS')
        paths['selected_frames_map']['aps'] = str(Path(args.selected).resolve())
    for key in ('annotation', 'reference_annotation', 'candidate_frames', 'utils'):
        if not Path(paths[key]).is_file():
            raise FileNotFoundError(paths[key])
    if not (model / 'config.json').is_file() or not list(model.glob('*.safetensors')):
        raise FileNotFoundError('Use a local model snapshot containing config.json and safetensors')
    return paths


def preflight(args):
    import torch
    import torchvision
    import decord
    from transformers import AutoProcessor
    paths = choose_paths(args)
    dscfg = DATASETS[args.dataset]
    docs = read_json(paths['annotation'])
    reference = read_json(paths['reference_annotation'])
    selections = {m: read_json(p) for m, p in paths['selected_frames_map'].items()}
    candidates = read_json(paths['candidate_frames'])
    for selected_path in paths['selected_frames_map'].values():
        meta_path = Path(selected_path).with_suffix('.meta.json')
        if not meta_path.is_file():
            raise ValueError('Selection metadata missing; generate with aps.select --annotation')
        meta = read_json(meta_path)
        for key, file in [('annotation_sha256', paths['annotation']),
                          ('frames_sha256', paths['candidate_frames']),
                          ('selected_sha256', selected_path)]:
            if meta.get(key) != sha(file):
                raise ValueError(f'Selection provenance mismatch: {key}')
    if dscfg['float_frames']:
        selections = {m: as_int_frames(s) for m, s in selections.items()}
        candidates = as_int_frames(candidates)
    if len(docs) != dscfg['n_questions'] or len(reference) != len(docs) or len(candidates) != len(docs) \
            or any(len(s) != len(docs) for s in selections.values()):
        raise ValueError(f'作者标注/下载标注/选帧/候选帧数量不匹配: '
                         f'{[len(docs),len(reference),len(candidates)]+[len(s) for s in selections.values()]}')
    if len({d[dscfg['id_key']] for d in docs}) != len(docs):
        raise ValueError('题目 ID 重复')
    annotation_differences = check_annotation_identity(docs, reference, dscfg)
    print('评测标注: 作者任务配置指定的', paths['annotation'], flush=True)
    if annotation_differences:
        print(f'下载的 {dscfg["reference"]} 与作者标注有 {len(annotation_differences)} 题内容不同；本次统一使用作者标注。', flush=True)
        print('差异示例:', annotation_differences[:10], flush=True)
    for doc in docs:
        video = Path(paths['video_root']) / (doc[dscfg['video_key']] + dscfg['video_suffix'])
        if not video.is_file() or video.stat().st_size == 0:
            raise FileNotFoundError(f'视频缺失或为空: {video}')
        if not dscfg['check_answer'](doc):
            raise ValueError(f'正确答案无效: {doc[dscfg["id_key"]]}')
    count = min(args.limit or len(docs), len(docs))
    indices = list(range(count))
    processor = AutoProcessor.from_pretrained(paths['model'], local_files_only=True,
                                              trust_remote_code=(paths['model_type'] in ('internvl', 'minicpmv')))
    functions = author_functions(paths['utils'], dscfg['function_names'])
    mcfg = MODEL_TYPES[paths['model_type']]
    nframes = args.nframes or mcfg['nframes']
    metadata = {}
    worst = None
    for done, i in enumerate(indices, 1):
        name = docs[i][dscfg['video_key']] + dscfg['video_suffix']
        if name not in metadata:
            vr = decord.VideoReader(str(Path(paths['video_root']) / name), ctx=decord.cpu(0), num_threads=1)
            height, width = vr[0].asnumpy().shape[:2]
            if paths['model_type'] == 'qwen2vl':
                rh, rw = native_resize(height, width)
            else:
                # InternVL / MiniCPM 的图像处理器使用各自的 frame_size。
                rh = rw = mcfg['frame_size']
            metadata[name] = {'total_frames': len(vr), 'fps': float(vr.get_avg_fps()),
                              'height': height, 'width': width, 'resized_height': rh, 'resized_width': rw}
            del vr
        info = metadata[name]
        if not math.isfinite(info['fps']) or int(info['fps']) < 1:
            raise ValueError(f'无法读取有效视频帧率: {name}, fps={info["fps"]}')
        if info['total_frames'] < nframes:
            raise ValueError(f'视频少于{nframes}原始帧，无法对齐作者设置: {name}')
        expected_candidates = list(range(0, (info['total_frames'] // int(info['fps'])) * int(info['fps']), int(info['fps'])))
        if candidates[i] != expected_candidates:
            raise ValueError(f'视频与作者 BLIP 缓存的帧网格不一致: {name}; 当前帧数={info["total_frames"]}, fps={info["fps"]:.5f}, 缓存候选帧数={len(candidates[i])}。不要继续使用错位缓存。')
        for m, sel in selections.items():
            validate_selected(sel[i], info['total_frames'])
            if len(sel[i]) != min(nframes, len(candidates[i])):
                raise ValueError('Selected frame count does not match the requested budget')
            if not set(sel[i]).issubset(set(candidates[i])):
                raise ValueError(f'选帧结果({m})不属于对应题目的候选帧: {docs[i][dscfg["id_key"]]}')
        text = functions[dscfg['doc_to_text']](docs[i], dscfg['prompt_kwargs'])
        prompt = build_prompt(paths['model_type'], processor, text)
        if paths['model_type'] == 'qwen2vl':
            vision_tokens = (nframes // 2) * (info['resized_height'] // 28) * (info['resized_width'] // 28)
        else:
            vision_tokens = mcfg['tokens_per_frame'] * nframes
        total_tokens = vision_tokens + len(processor.tokenizer.encode(prompt) if hasattr(processor, 'tokenizer') else processor.encode(prompt)) + dscfg['max_tokens']
        if total_tokens > args.max_model_len:
            raise ValueError(f'{name}: 预计上下文 {total_tokens} 超过 {args.max_model_len}。脚本不会自动降分辨率改变评测设置。')
        if worst is None or vision_tokens > worst[0]:
            worst = (vision_tokens, info['resized_height'], info['resized_width'], name)
        if done % 100 == 0:
            print(f'视频/帧编号检查 {done}/{count}', flush=True)
    print(f'视觉token峰值(单prompt): {worst[0]} @ {worst[3]}', flush=True)
    config = {**paths, 'indices': indices, 'gpus': args.gpus.split(';'),
              'methods': args.methods.split(','),
              'group_fields': list(dscfg['group_fields']),
              'prompt_kwargs': dscfg['prompt_kwargs'],
              'nframes': nframes, 'max_model_len': args.max_model_len, 'seed': args.seed,
              'kv_cache_gib': args.kv_cache_gib, 'metadata': metadata,
              'annotation_differences': annotation_differences,
              'profile_height': worst[1], 'profile_width': worst[2],
              'post_prompt': dscfg['prompt_kwargs']['post_prompt'], 'max_tokens': dscfg['max_tokens'],
              'dtype': 'bfloat16', 'tensor_parallel_size': args.tensor_parallel_size,
              'enforce_eager': True, 'max_num_seqs': 1, 'max_num_batched_tokens': 6274,
              'versions': {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'transformers', 'vllm', 'decord')},
              'hashes': {**{key: sha(paths[key]) for key in ('annotation','reference_annotation','candidate_frames','utils')},
                         **{f'selected_frames:{m}': sha(p) for m, p in paths['selected_frames_map'].items()}},
              'model_config_hash': sha(Path(paths['model']) / 'config.json'),
              'processor_config_hash': sha(Path(paths['model']) / 'preprocessor_config.json')
                                        if (Path(paths['model']) / 'preprocessor_config.json').is_file() else '',
              'runner_hash': sha(__file__),
              'weights': {p.name: [p.stat().st_size,p.stat().st_mtime_ns] for p in sorted(Path(paths['model']).glob('*.safetensors'))}}
    print(f'检查通过: {args.dataset} 验证集{len(docs)}题; 本次{count}题; 原生分辨率预处理; 无字幕; {nframes}帧; 方法: {args.methods}', flush=True)
    print('每卡题数:', [len(assigned(indices,r,len(config['gpus']))) for r in range(len(config['gpus']))], flush=True)
    return config


def worker(run_dir, rank):
    import torch
    import decord
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    torch.set_num_threads(2)
    cfg = read_json(Path(run_dir) / 'manifest.json')
    dscfg = DATASETS[cfg.get('dataset', 'longvideobench')]
    docs = read_json(cfg['annotation'])
    functions = author_functions(cfg['utils'], dscfg['function_names'])
    indices = assigned(cfg['indices'], rank, len(cfg['gpus']))
    methods_all = cfg['methods']
    sel_paths = cfg['selected_frames_map']
    selections = {m: read_json(p) for m, p in sel_paths.items() if m in methods_all}
    candidates = read_json(cfg['candidate_frames']) if selections else None
    if dscfg['float_frames']:
        selections = {m: as_int_frames(s) for m, s in selections.items()}
        if candidates is not None:
            candidates = as_int_frames(candidates)
    # Read only this worker's result file: other workers may be appending theirs.
    done = load_results(run_dir, rank=rank)
    pending = [(i,m) for i in indices for m in methods_all if (i,m) not in done]
    if not pending:
        print(f'GPU {cfg["gpus"][rank]} 已完成本分片，无需加载模型', flush=True)
        return
    print(f'GPU {cfg["gpus"][rank]}: {len(indices)}道题，待推理{len(pending)}次', flush=True)
    if torch.cuda.device_count() != cfg['tensor_parallel_size']:
        raise RuntimeError('Visible GPUs do not match tensor_parallel_size')
    processor = AutoProcessor.from_pretrained(cfg['model'], local_files_only=True,
                                              trust_remote_code=(cfg.get('model_type') in ('internvl', 'minicpmv')))
    model_type = cfg.get('model_type', 'qwen2vl')
    mcfg = MODEL_TYPES[model_type]
    nframes = cfg.get('nframes', mcfg['nframes'])
    if model_type == 'internvl':
        mm_kwargs = {}  # InternVLImageProcessor 不接受 do_sample_frames/do_resize 等参数
    elif model_type == 'minicpmv':
        mm_kwargs = {}  # MiniCPM 图像处理器也不接受 do_sample_frames
    elif model_type == 'qwen2vl':
        mm_kwargs = {'do_resize': False, 'do_sample_frames': False,
                     # 仅影响 vLLM 启动时的 encoder cache 预算推导
                     # （_MAX_FRAMES_PER_VIDEO=14 → 7×max_pixels/784 = 126700 token）；
                     # do_resize=False 使实际推理仍用原生分辨率，不改变评测协议。
                     # MLVU 竖屏视频原生 32 帧峰值 126464 token，默认预算 114688 会被拒；
                     # cells 上限受 max_model_len=127800 约束（7×cells ≤ 127800）。
                     'max_pixels': 14190400}
    else:
        raise ValueError(f'Unsupported answering model type: {model_type}')
    trc = model_type in ('internvl', 'minicpmv')  # 这些仓库带 trust_remote_code 自定义代码
    if model_type == 'minicpmv':
        # MiniCPM 视频按 16 张图像喂，不走 vllm 的视频通道
        limit_mm = {'image': nframes, 'video': 0}
    else:
        limit_mm = {'image': 0, 'video': {'count': 1, 'num_frames': nframes,
                    'height': cfg['profile_height'], 'width': cfg['profile_width']}}
    llm = LLM(model=cfg['model'], dtype='bfloat16', tensor_parallel_size=cfg['tensor_parallel_size'],
              max_model_len=cfg['max_model_len'], max_num_seqs=1,
              max_num_batched_tokens=6274, enforce_eager=True,
              enable_chunked_prefill=True, enable_prefix_caching=False,
              kv_cache_memory_bytes=int(cfg['kv_cache_gib'] * 1024**3),
              mm_processor_kwargs=mm_kwargs,
              trust_remote_code=trc,
              limit_mm_per_prompt=limit_mm,
              seed=cfg['seed'])
    sampling = SamplingParams(temperature=0, max_tokens=cfg.get('max_tokens', 32))
    path = Path(run_dir) / f'rank{rank}.jsonl'
    requests = 0
    with path.open('a', buffering=1) as out:
        for i in indices:
            methods = [m for m in methods_all if (i,m) not in done]
            if not methods:
                continue
            doc = docs[i]
            name = doc[dscfg['video_key']] + dscfg['video_suffix']
            video_path = Path(cfg['video_root']) / name
            try:
                vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
                if len(vr) < nframes:
                    raise ValueError(f'视频少于{nframes}原始帧，无法对齐作者nframes设置')
                if model_type in ('internvl', 'minicpmv'):
                    # InternVL/MiniCPM 居中分段采样（[0, n-1] 分 nframes 段取每段中点）
                    seg = len(vr) / nframes
                    uniform = [int(seg / 2 + round(seg * k)) for k in range(nframes)]
                else:
                    uniform = torch.linspace(0, len(vr)-1, nframes).round().long().tolist()
                info = cfg['metadata'][name]
                text = functions[dscfg['doc_to_text']](doc, cfg.get('prompt_kwargs', dscfg['prompt_kwargs']))
                if model_type == 'minicpmv':
                    # MiniCPM 视频：16 张图像 + 空格分隔占位符（BPE 合词防护）
                    img_tokens = ' '.join(['(<image>./</image>)'] * nframes)
                    prompt = (f'<|im_start|>user\n {img_tokens} {text}<|im_end|>\n'
                              f'<|im_start|>assistant\n')
                else:
                    prompt = build_prompt(model_type, processor, text)
                for method in methods:
                    start = time.perf_counter()
                    if method == 'uniform':
                        fallback, frame_idx = False, uniform
                    else:
                        sel_i = selections[method][i]
                        # 候选不足预算时，以整段视频的均匀采样替换整组输入。
                        fallback = len(sel_i) < nframes and len(candidates[i]) < nframes
                        frame_idx = uniform if fallback else sorted(sel_i[:nframes])
                    # decord 在个别视频尾部取帧偶发 EOF 失败（同一视频其他分片可读），
                    # 重建 VideoReader 重试，避免整轮评测被一道题拖死
                    for decode_attempt in range(4):
                        try:
                            raw = vr.get_batch(frame_idx).asnumpy()
                            break
                        except decord.DECORDError:
                            if decode_attempt == 3:
                                raise
                            print(f'rank={rank} index={i} method={method}: 解码失败，'
                                  f'重建reader重试({decode_attempt+1}/3)', flush=True)
                            del vr
                            time.sleep(5 * (decode_attempt + 1))
                            vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
                    if model_type == 'internvl':
                        # InternVL 由 vLLM 的 InternVLVideoProcessor 做 448 单 tile，
                        # 需传 uint8 原始帧（PIL.fromarray 不接受 float32）
                        frames = raw
                    elif model_type == 'minicpmv':
                        # MiniCPM：uint8 原始帧转 PIL 图像列表（由官方 processor 动态切 tile）
                        from PIL import Image
                        frames = [Image.fromarray(raw[k]) for k in range(len(raw))]
                    else:
                        video = torch.tensor(raw).permute(0,3,1,2)
                        video = resize(video, [info['resized_height'],info['resized_width']],
                                       interpolation=InterpolationMode.BICUBIC, antialias=True).float()
                        frames = video.permute(0,2,3,1).contiguous().numpy()
                    mm_data = {'image': frames} if model_type == 'minicpmv' else {'video': frames}
                    response = llm.generate([{'prompt':prompt,'multi_modal_data':mm_data}],
                                            sampling, use_tqdm=False)[0].outputs[0].text
                    if dscfg['random_parser']:
                        mapping, choices = functions['get_multi_choice_info'](doc['candidates'])
                        # Author parser uses a random fallback when no letter is found.
                        # Seed per question, making it stable across sharding/resume.
                        random.seed(cfg['seed'] + i)
                        before = random.getstate()
                        pred = functions['parse_multi_choice_response'](response, choices, mapping)
                        unparsed = random.getstate() != before
                    else:
                        # extract_characters_regex 是纯函数，无随机兜底；解析失败返回空串
                        pred = functions['extract_characters_regex'](response)
                        unparsed = (pred == '')
                    gold = dscfg['gold_of'](doc)
                    record = {'status':'ok','index':i,'id':doc[dscfg['id_key']],'method':method,
                              'video_path':name,'frame_indices':frame_idx,
                              'uniform_frame_fallback':fallback,'response':response,
                              'prediction':pred,'answer':gold,
                              'correct':pred==gold,
                              'unparsed_fallback':unparsed,
                              **{f: doc[f] for f in dscfg['group_fields']},
                              'seconds_including_decode':time.perf_counter()-start,
                              'first_request_of_worker':requests==0}
                    out.write(json.dumps(record,ensure_ascii=False)+'\n')
                    requests += 1
                    done[(i,method)] = record
                    print(f"rank={rank} index={i} method={method} pred={pred} gold={record['answer']} correct={record['correct']} response={response!r}", flush=True)
                    del frames
                del vr
            except Exception as exc:
                out.write(json.dumps({'status':'error','index':i,'id':doc[dscfg['id_key']],'error':repr(exc)},ensure_ascii=False)+'\n')
                raise
    print(f'分片 rank={rank} 完成', flush=True)


def stop_children(children):
    for proc in children:
        try:
            # Includes vLLM engine children even if the outer worker has exited.
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic()+15
    while any(p.poll() is None for p in children) and time.monotonic()<deadline:
        time.sleep(0.2)
    for proc in children:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def launch(args):
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / 'run.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'此结果目录已有任务运行: {run_dir}')
    cfg = preflight(args)
    manifest = run_dir / 'manifest.json'
    if manifest.exists():
        if read_json(manifest) != cfg:
            raise ValueError('此目录的参数/数据/代码/模型与上次不同。使用新的 --run-dir，避免混合结果。')
    else:
        if list(run_dir.glob('rank*.jsonl')):
            raise ValueError('结果目录有预测但没有manifest，不能安全续跑')
        write_json(manifest,cfg)
    existing = load_results(run_dir, repair_tail=True)
    expected = {(i,m) for i in cfg['indices'] for m in cfg['methods']}
    if not set(existing).issubset(expected):
        raise ValueError('结果目录包含本次评测范围之外的预测')
    if set(existing) == expected:
        return summarize(run_dir)
    children, handles = [], []
    def stop_on_term(signum, frame):
        raise KeyboardInterrupt
    old_term_handler = signal.signal(signal.SIGTERM, stop_on_term)
    try:
        for rank,gpu in enumerate(cfg['gpus']):
            env = os.environ.copy()
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
                env.pop(key,None)
            env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       VLLM_WORKER_MULTIPROC_METHOD='spawn', DECORD_EOF_RETRY_MAX='40960')
            env.pop('VLLM_PLUGINS', None)
            handle = (run_dir / f'rank{rank}.log').open('a')
            handles.append(handle)
            proc = subprocess.Popen([sys.executable,'-u','-m','aps.infer',
                                      '_worker','--run-dir',str(run_dir),'--rank',str(rank)],
                                     env=env, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            children.append(proc)
            print(f'已启动 GPU={gpu} rank={rank} pid={proc.pid}; 日志={run_dir / f"rank{rank}.log"}',flush=True)
        last = 0
        while any(p.poll() is None for p in children):
            for rank,p in enumerate(children):
                if p.poll() not in (None,0):
                    raise RuntimeError(f'rank={rank} 退出码={p.returncode}，请查看 {run_dir / f"rank{rank}.log"}；已保存的预测可续跑')
            if time.monotonic()-last>=30:
                count = sum(p.read_bytes().count(b'\n') for p in run_dir.glob('rank*.jsonl'))
                print(f'预测记录 {count}/{len(expected)}; 完成进程 {sum(p.poll()==0 for p in children)}/{len(children)}',flush=True)
                last=time.monotonic()
            time.sleep(1)
        failed = [r for r,p in enumerate(children) if p.returncode != 0]
        if failed:
            raise RuntimeError(f'失败分片: {failed}')
        summarize(run_dir)
    finally:
        stop_children(children)
        for handle in handles:
            handle.close()
        signal.signal(signal.SIGTERM, old_term_handler)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'run', 'summarize', '_worker'))
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--model', default='models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--dataset', choices=tuple(DATASETS), default='longvideobench')
    parser.add_argument('--annotation', default='')
    parser.add_argument('--reference', default='')
    parser.add_argument('--video-root', default='')
    parser.add_argument('--candidates', default='')
    parser.add_argument('--selected', default='')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--gpus', default='0', help='DP groups: 0;1;2;3. TP groups: 0,1;2,3 (quote the string)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--nframes', type=int, default=0, help='0: model default (32/32/16)')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--max-model-len', type=int, default=32768)
    parser.add_argument('--kv-cache-gib', type=float, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--methods', default='aps', help='APS by default; optionally include the Uniform control')
    args = parser.parse_args()
    groups = [g.split(',') for g in args.gpus.split(';')]
    gpus = [x for g in groups for x in g]
    if args.tensor_parallel_size < 1 or any(len(g) != args.tensor_parallel_size for g in groups):
        parser.error('Each GPU group must match --tensor-parallel-size')
    if any(not x.isdigit() for x in gpus) or len(gpus) != len(set(gpus)):
        parser.error('GPU indices must be numeric and unique')
    if args.limit < 0 or args.nframes < 0 or (args.nframes and args.nframes % 2):
        parser.error('limit must be nonnegative; nframes must be zero or positive/even')
    methods = args.methods.split(',')
    if not methods or len(methods) != len(set(methods)) or any(m not in METHODS for m in methods):
        parser.error('methods must be a nonempty subset of aps,uniform')
    if args.max_model_len <= 0 or not math.isfinite(args.kv_cache_gib) or args.kv_cache_gib <= 0:
        parser.error('Invalid context length or KV-cache size')
    if args.command == '_worker':
        worker(args.run_dir, args.rank)
    elif args.command == 'summarize':
        summarize(args.run_dir)
    elif args.command == 'check':
        preflight(args)
    else:
        launch(args)


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('已中断，已完成预测保留；相同命令可继续。',file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
