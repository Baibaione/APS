"""Small immutable-manifest and JSONL helpers."""
import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def manifest(path, data):
    path = Path(path)
    if path.exists():
        if read(path) != data:
            raise ValueError(f"Input/configuration changed; use a new output directory: {path}")
    else:
        write(path, data)


def jsonl(path, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        if not repair:
            raise ValueError(f"Incomplete final JSONL line: {path}")
        data = data[:data.rfind(b"\n") + 1]
        path.write_bytes(data)
    return [json.loads(line) for line in data.splitlines() if line.strip()]
