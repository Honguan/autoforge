# 發佈流程

推送 `v1.0.0` 這類版本 tag 時，`.github/workflows/release.yml` 會自動執行：

1. 檢查 tag 和 `pyproject.toml` 的版本是否一致。
2. 建置 wheel 和 sdist，執行 `twine check`。
3. 在全新的 venv 安裝 wheel，並跑一次 `autoforge quickstart`。
4. 發佈到 **PyPI**（Trusted Publishing，不需要存放 token）。
5. 建立 **GitHub Release**，附上 dist 檔案和 CHANGELOG。
6. 建置並推送 **Docker 映像**到 GHCR：`ghcr.io/honguan/autoforge:cpu`、`:cuda`，以及帶版本號的 tag。

推送到 `main` 且 `docs/` 有變動時，`docs.yml` 會更新文件網站。

## 第一次發佈前需要做的設定（只做一次）

| 設定 | 步驟 |
|---|---|
| PyPI Trusted Publishing | 登入 pypi.org → Your account → Publishing → **Add a new pending publisher**：PyPI project name `autoforge-ml`、Owner `Honguan`、Repository `autoforge`、Workflow `release.yml`、Environment `pypi` |
| GitHub 環境 | repo Settings → Environments → New environment，名稱填 `pypi`（建議設 Required reviewers，發佈前要你按同意） |
| GitHub Pages | repo Settings → Pages → Source 選 **GitHub Actions** |
| GHCR 映像公開 | 第一次推送後：GitHub 個人頁 → Packages → `autoforge` → Package settings → Change visibility → Public |

## 發佈一個版本

先改 `pyproject.toml` 和 `autoforge/__init__.py` 的版本號，更新 CHANGELOG.md，合併到 main。接著用下面任一種方式建立 tag：

**方法 A：命令列**

```bash
git checkout main && git pull
git tag v1.0.0
git push origin v1.0.0
```

**方法 B：GitHub 網頁**

1. 到 repo 首頁 → Releases → **Draft a new release**。
2. Choose a tag 輸入 `v1.0.0`，選「Create new tag on publish」，Target 選 `main`。
3. 標題填 `AutoForge v1.0.0`，按 **Publish release**。

建立 tag 後就會觸發 `release` workflow，自動把 wheel 和 sdist 附加到這個 Release。

接著到 GitHub 的 Actions 頁面看 `release` 的執行結果。

## 在本機驗證「乾淨機器」

```bash
python -m build
docker run --rm -v $PWD/dist:/dist python:3.12-slim \
  bash -c 'pip install /dist/*.whl && time autoforge quickstart'
```
