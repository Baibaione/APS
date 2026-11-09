"""Select APS frames from question-aligned score/frame caches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from .io import manifest, read, sha, write
from .selector import APSConfig, select_frames


def one(item):
    scores, frames, cfg = item
    return select_frames(scores, frames, cfg, return_details=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--annotation", help="Bind selections to a precise question order and annotation snapshot")
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, required=True, help="Main table: LVB=.6; VideoMME/MLVU=.4")
    p.add_argument("--eta", "--penalty", dest="penalty", type=float, default=.5,
                   help="Complexity regularization strength eta (paper Eq. 8)")
    p.add_argument("--variant", default="full", choices=["full", "equal_length", "linear_only", "constant_only"])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--details", action="store_true", help="Save per-question intervals and allocations")
    a = p.parse_args()
    if a.workers < 1:
        p.error("workers must be positive")
    cfg = APSConfig(a.budget, a.alpha, a.beta, a.penalty, a.variant)
    scores, frames = read(a.scores), read(a.frames)
    if not scores or len(scores) != len(frames):
        raise ValueError("Empty or mismatched score/frame cache")
    dest = Path(a.output)
    ident = {"method": "aps", "config": asdict(cfg), "questions": len(scores),
             "scores_sha256": sha(a.scores), "frames_sha256": sha(a.frames),
             "selector_sha256": sha(Path(__file__).with_name("selector.py")),
             "kernels_sha256": sha(Path(__file__).with_name("kernels.py"))}
    if a.annotation:
        if len(read(a.annotation)) != len(scores):
            raise ValueError("Annotation/score question counts differ")
        ident["annotation_sha256"] = sha(a.annotation)
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest(dest.with_suffix(".inputs.json"), ident)
    items = ((s, f, cfg) for s, f in zip(scores, frames))
    if a.workers == 1:
        rows = list(map(one, items))
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            rows = list(pool.map(one, items, chunksize=8))
    selected = [r[0] for r in rows]
    if dest.exists() and read(dest) != selected:
        raise ValueError("Existing selected frames differ; use a new output path")
    write(dest, selected)
    write(dest.with_suffix(".meta.json"), {**ident, "selected_sha256": sha(dest),
          "uniform_fallback_questions": sum(r[1]["uniform_fallback"] for r in rows)})
    if a.details:
        write(dest.with_suffix(".details.json"), [r[1] for r in rows])
    print(f"Saved {len(rows)} question selections to {dest}")


if __name__ == "__main__":
    main()
