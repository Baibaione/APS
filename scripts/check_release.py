#!/usr/bin/env python3
"""Check the allowlisted release tree for private paths, credentials and blobs."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git','__pycache__','.pytest_cache','.ruff_cache','build','dist','data','cache','models','outputs','.venv'}
patterns = [re.compile(x) for x in [r'QC-[a-f0-9]{24,}', r'sk-[A-Za-z0-9_-]{20,}',
             r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
             r'/public' + r'-nfs/', r'/Users/' + r'[^/\s]+/',
             r'192\.168\.\d+\.\d+',
             r'''(?i)(?:api_key|api_token|password)\s*[:=]\s*["'][^"']{8,}["']''']]
errors=[];count=0
for p in ROOT.rglob('*'):
    relative=p.relative_to(ROOT)
    if any(x in SKIP or x.endswith('.egg-info') for x in relative.parts) or not p.is_file():
        continue
    count+=1
    if p.is_symlink(): errors.append((str(relative),'symlink'))
    if p.stat().st_size > 10*1024**2: errors.append((str(relative),'over 10 MiB'))
    if p.suffix in {'.safetensors','.pth','.pt','.mp4','.mkv','.pem','.key','.log'}:
        errors.append((str(relative),'private/runtime artifact'))
    try:text=p.read_text()
    except UnicodeDecodeError:continue
    if any(x.search(text) for x in patterns):errors.append((str(relative),'sensitive pattern'))
if errors:
    for p,why in errors:print(p,why)
    raise SystemExit(1)
print(f'Release check passed: {count} files; no detected credential/private-path patterns.')
