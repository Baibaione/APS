"""Question/frame scoring with BLIP, CLIP or BLIP-2 on CUDA.

Cache reuse is preferred for reference selection reproduction; see docs/data.md.
No answer field is passed to a scorer. MLVU's question already contains options.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from .io import jsonl, manifest, read, sha, write


def pre_caption(caption):
    caption = re.sub(r'([.!"()*#:;~])', ' ', caption.lower())
    caption = re.sub(r'\s{2,}', ' ', caption).rstrip('\n').strip(' ')
    return ' '.join(caption.split(' ')[:50])


def video_name(doc, dataset):
    if dataset == 'longvideobench':
        return doc['video_path']
    if dataset == 'videomme':
        return doc['videoID'] + '.mp4'
    return doc['video_relpath']


def candidate_grid(total, fps):
    if total < 1 or not np.isfinite(fps) or int(fps) < 1:
        raise ValueError('Invalid video length or FPS')
    stride = int(fps)
    grid = list(range(0, (total // stride) * stride, stride))
    if not grid:
        raise ValueError('Video has no candidate frames under the candidate sampling grid')
    return grid


class Scorer:
    def __init__(self, name, model_path):
        import torch
        from transformers import AutoProcessor, BertTokenizerFast, BlipForImageTextRetrieval, CLIPModel, Blip2ForImageTextRetrieval
        self.name, self.torch = name, torch
        cls = {'blip': BlipForImageTextRetrieval, 'clip': CLIPModel, 'blip2': Blip2ForImageTextRetrieval}[name]
        self.model = cls.from_pretrained(model_path, torch_dtype=torch.float32).eval().cuda()
        if name == 'blip':
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)

    def __call__(self, raw, question):
        from PIL import Image
        torch = self.torch
        images = [Image.fromarray(x) for x in raw]
        with torch.inference_mode():
            if self.name == 'blip':
                pixels = torch.stack([torch.from_numpy(np.array(x.resize((384, 384), Image.Resampling.BICUBIC))).permute(2, 0, 1) for x in images]).float() / 255
                mean = torch.tensor((.48145466, .4578275, .40821073)).view(1, 3, 1, 1)
                std = torch.tensor((.26862954, .26130258, .27577711)).view(1, 3, 1, 1)
                pixels = ((pixels - mean) / std).cuda()
                tok = self.tokenizer(pre_caption(question), padding='max_length', truncation=True, max_length=35, return_tensors='pt')
                tok = {k: v.expand(len(images), -1).cuda() for k, v in tok.items() if k in ('input_ids', 'attention_mask')}
                logits = self.model(pixel_values=pixels, **tok).itm_score
                values = logits.softmax(dim=1)[:, 1]
            elif self.name == 'clip':
                pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].cuda()
                tok = self.processor(text=question, padding=True, truncation=True, return_tensors='pt')
                text = self.model.get_text_features(**{k: v.cuda() for k, v in tok.items()})
                vision = self.model.get_image_features(pixel_values=pixels)
                values = torch.nn.functional.cosine_similarity(vision, text, dim=-1)
            else:
                pixels = self.processor.image_processor(images=images, return_tensors='pt')['pixel_values'].cuda()
                tok = self.processor.tokenizer(question, truncation=True, max_length=32, return_tensors='pt')
                tok = {k: v.expand(len(images), -1).cuda() for k, v in tok.items() if k in ('input_ids', 'attention_mask')}
                logits = self.model(pixel_values=pixels, **tok, use_image_text_matching_head=True).logits_per_image
                values = logits.float().softmax(dim=1)[:, 1]
        result = values.float().cpu().tolist()
        if not np.isfinite(result).all():
            raise ValueError('Scorer returned non-finite values')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['longvideobench', 'videomme', 'mlvu'])
    p.add_argument('--annotation', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--scorer', choices=['blip', 'clip', 'blip2'], default='blip')
    p.add_argument('--candidates', help='Reuse the same frame grid for encoder ablations')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--merge', action='store_true')
    a = p.parse_args()
    if not 0 <= a.shard_id < a.num_shards or a.batch_size < 1:
        p.error('Invalid shard or batch size')
    docs = read(a.annotation)
    dest = Path(a.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    # One common identity for all shards. Caller must not run a shard twice concurrently.
    import fcntl
    with (dest / 'manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = Path(a.model).resolve()
        ident = dict(dataset=a.dataset, annotation_sha256=sha(a.annotation),
                     model=str(model), scorer=a.scorer, num_shards=a.num_shards,
                     video_root=str(Path(a.video_root).resolve()),
                     candidates_sha256=sha(a.candidates) if a.candidates else None,
                     source_sha256=sha(__file__), batch_size=a.batch_size,
                     model_config_sha256=sha(model/'config.json'),
                     weights={x.name: [x.stat().st_size, x.stat().st_mtime_ns] for x in sorted(model.glob('*.safetensors'))})
        if not ident['weights']:
            raise ValueError('Model snapshot has no safetensors weights')
        manifest(dest / 'manifest.json', ident)
    if a.merge:
        rows = {}
        for rank in range(a.num_shards):
            for row in jsonl(dest / f'rank{rank}.jsonl'):
                i = row['index']
                if i in rows or i % a.num_shards != rank:
                    raise ValueError('Duplicate or misplaced scoring record')
                rows[i] = row
        if set(rows) != set(range(len(docs))):
            raise ValueError(f'Scoring incomplete: {len(rows)}/{len(docs)}')
        for name in ['scores', 'frames']:
            write(dest / (name + '.json'), [rows[i][name] for i in range(len(docs))])
        write(dest / 'complete.json', dict(questions=len(docs), scores_sha256=sha(dest/'scores.json'), frames_sha256=sha(dest/'frames.json')))
        print('Merged', len(rows), 'questions')
        return
    with (dest / f'rank{a.shard_id}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = dest / f'rank{a.shard_id}.jsonl'
        old = jsonl(path, repair=True)
        done = {r['index'] for r in old}
        if len(done) != len(old) or any(i % a.num_shards != a.shard_id for i in done):
            raise ValueError('Invalid scoring checkpoint')
        pending = [i for i in range(len(docs)) if i % a.num_shards == a.shard_id and i not in done]
        if not pending:
            print('Scoring shard already complete')
            return
        import decord
        scorer = Scorer(a.scorer, a.model)
        supplied = read(a.candidates) if a.candidates else None
        if supplied is not None and len(supplied) != len(docs):
            raise ValueError('Candidate grid/question count mismatch')
        with path.open('a', buffering=1) as f:
            for i in pending:
                vr = decord.VideoReader(str(Path(a.video_root) / video_name(docs[i], a.dataset)), num_threads=2)
                grid = candidate_grid(len(vr), vr.get_avg_fps())
                if supplied is not None and supplied[i] != grid:
                    raise ValueError('Candidate cache does not match decoded video')
                values = []
                for start in range(0, len(grid), a.batch_size):
                    raw = vr.get_batch(grid[start:start+a.batch_size]).asnumpy()
                    values += scorer(raw, docs[i]['question'])
                if a.dataset == 'mlvu' and a.scorer == 'blip':
                    values = [round(x, 6) for x in values]
                f.write(json.dumps(dict(index=i, frames=grid, scores=values), allow_nan=False) + '\n')
                print(f'shard={a.shard_id} index={i} candidates={len(grid)}', flush=True)


if __name__ == '__main__':
    main()
