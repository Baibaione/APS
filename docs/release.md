# Local repository and release preparation

This repository is separate from the original server experiment directory. Do not run `git add .` in that old directory: it may contain models, data, machine settings, and runtime credentials.

Versioned content includes APS numerical kernels, controlled ablations, scoring/inference entry points, evaluation protocols, configurations, reference selections, question-identity hashes, sanitized aggregate metrics, tests, and documentation. `README.md` is the English entry point; `README.zh-CN.md` provides the Chinese version. Detailed documentation and code comments use English.

The method scope is APS. Uniform is retained as an ablation control and short-video fallback. Main-table configurations run APS by default. Implementations, integration frameworks, run configurations, and aggregate results for other selection methods are excluded. Historical data references and required third-party licenses are retained.

Excluded content includes model weights, full datasets, videos and case-study screenshots, large BLIP caches, raw responses, machine IP addresses, SSH configuration, API keys, container snapshots, and retry logs. A separate reproduction-input archive may be kept outside the repository as a personal backup. Any future release of that archive must follow the source data's licenses.

Run local checks:

```bash
python -m pip install -e '.[test]'
pytest -q
ruff check src scripts tests
python scripts/check_release.py
python -m build
```

Initialize and inspect the files to be committed. If the repository is already initialized, start with `git status`:

```bash
git init -b main
git status --short
git add README.md README.zh-CN.md pyproject.toml MANIFEST.in requirements-nvidia*.txt .gitignore .github src configs assets reports scripts examples tests docs third_party
git diff --cached --stat
git diff --cached --check
git commit -m "Prepare APS research code and reproduction protocol"
```

After creating an empty **private** GitHub repository, use its URL:

```bash
git remote add origin <YOUR_PRIVATE_REPOSITORY_URL>
git push -u origin main
```

Keep the remote private until a public release is intended.

Items remaining before public release:

1. Supply the final paper title, authors, and BibTeX entry; no placeholder DOI or invented citation is included.
2. Choose a project license for the original code. Only third-party licenses are currently retained; no MIT/Apache grant for APS is assumed.
3. Run small GPU smoke tests for all nine model/benchmark combinations on NVIDIA hardware. Record the GPU, driver, dependencies, and memory usage, then pin the validated environment.
4. Distinguish historical MetaX results from later NVIDIA measurements, particularly for inference efficiency.

`check_release.py` checks the release directory but is not a comprehensive secret scanner. It does not print matched sensitive content. Review the staged diff before making the repository public.
