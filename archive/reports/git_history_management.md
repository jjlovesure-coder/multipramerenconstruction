# Git 历史版本管理说明

这份说明用于在 Codex、Cursor 或 PowerShell 中管理本仓库的 Git 历史版本。

## 当前分支和远端

当前工作分支：

```text
Richard
```

远端仓库：

```text
https://github.com/jjlovesure-coder/multipramerenconstruction.git
```

常用推送命令：

```powershell
git -c http.proxy= -c https.proxy= push origin Richard
```

## 快速查看历史

查看最近提交：

```powershell
git log --oneline --decorate --graph -12
```

查看某个提交改了什么：

```powershell
git show --stat <commit>
git show --name-status <commit>
```

对比两个提交：

```powershell
git diff <old_commit>..<new_commit> --stat
git diff <old_commit>..<new_commit>
```

查看当前工作区：

```powershell
git status -sb
git diff --stat
git diff --cached --stat
```

## 安全回退策略

优先使用非破坏性命令。

回退一个提交但保留修改在工作区：

```powershell
git reset --soft HEAD~1
```

撤销某个已经推送的提交，生成一个新的反向提交：

```powershell
git revert <commit>
```

查看旧版本文件内容，不改当前工作区：

```powershell
git show <commit>:path/to/file
```

从旧版本恢复单个文件时，先确认这是你想要的操作：

```powershell
git restore --source=<commit> -- path/to/file
```

不要随意使用：

```powershell
git reset --hard
git clean -fd
```

这些命令会丢弃本地修改，只有在明确确认后再用。

## Codex/Cursor 协作流程

推荐流程：

1. 查看状态：

```powershell
git status -sb
```

2. 查看最近历史：

```powershell
git log --oneline --decorate --graph -12
```

3. 检查本次变化：

```powershell
git diff --stat
git diff
```

4. 显式暂存文件：

```powershell
git add README.md docs/git_history_management.md
```

5. 提交：

```powershell
git commit -m "Add Git history management docs"
```

6. 推送：

```powershell
git -c http.proxy= -c https.proxy= push origin Richard
```

## 本仓库的注意事项

- 原始 DSC Excel 文件和生成的 `.mtd` 文件会被版本化，因为它们是实验记录的一部分。
- `results/` 中的模型指标、图表和 CSV 如果用于报告结论，可以版本化。
- `__pycache__/` 和 `.pyc` 不应版本化。
- 提交大文件前先确认它是实验记录所必需的。
- 如果要重写已经推送的历史，先备份分支，不要直接 force push。

## 辅助脚本

本仓库提供 PowerShell 辅助脚本：

```powershell
powershell -ExecutionPolicy Bypass -File tools/git_history.ps1 summary
powershell -ExecutionPolicy Bypass -File tools/git_history.ps1 recent
powershell -ExecutionPolicy Bypass -File tools/git_history.ps1 caches
```
