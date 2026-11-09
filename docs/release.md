# 本地仓库与后续发布

代码整理目录已与旧服务器实验目录分离。不要对旧目录直接执行 `git add .`，其中可能包含模型、数据、机器配置和运行凭据。

本目录提交内容：APS 数值内核、受控消融、打分/推理入口、评测协议、配置、正式选帧、必要的题目身份哈希、脱敏聚合指标、测试与说明。

方法范围限定为 APS；Uniform 作为消融对照和短视频回退保留。主表配置默认只运行 APS，不收录其他选帧方法的实现、接入框架、运行配置或结果汇总。历史数据来源引用和必要的第三方许可仍予保留。

不提交：模型权重、完整数据集、视频/案例帧截图、BLIP 大缓存、原始响应、机器 IP、SSH 配置、API Key、容器快照、失败重试日志。独立复现输入包位于仓库外，可作为个人备份；后续是否作为 Release 资源发布，应按数据来源许可处理。

先本地检查：

```bash
python -m pip install -e '.[test]'
pytest -q
ruff check src scripts tests
python scripts/check_release.py
python -m build
```

初始化并查看待提交文件（若已经初始化则直接从 `git status` 开始）：

```bash
git init -b main
git status --short
git add README.md pyproject.toml MANIFEST.in requirements-nvidia*.txt .gitignore .github src configs assets reports scripts examples tests docs third_party
git diff --cached --stat
git diff --cached --check
git commit -m "Prepare APS research code and reproduction protocol"
```

在 GitHub 建立空的**私有仓库**后，使用你自己的仓库地址：

```bash
git remote add origin <YOUR_PRIVATE_REPOSITORY_URL>
git push -u origin main
```

远端仓库保持 Private；正式开源时再单独调整可见性。

公开前的未定项目：

1. 填入最终论文标题、作者和正式 BibTeX；没有使用虚构题目或 DOI。
2. 为自有代码确定项目 LICENSE；当前只保留第三方许可，不假定你已授权采用 MIT/Apache。
3. 在目标 NVIDIA GPU 上运行九组小样本 smoke，记录 GPU、驱动、依赖和显存；把成功验证的环境锁定后再宣称支持。
4. 明确区分历史 MetaX 测量和后来 NVIDIA 测量，尤其是推理效率。

`check_release.py` 是目录体检，不是完整密钥扫描器；它不输出匹配到的敏感内容。实际 GitHub 公开前仍应查看待提交 diff。
