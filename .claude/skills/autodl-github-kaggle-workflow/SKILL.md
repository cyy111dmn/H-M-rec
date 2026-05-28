---
name: autodl-github-kaggle-workflow
description: Use this skill when working on H&M recommendation experiments that use local development, GitHub code sync, AutoDL remote execution, protected data directories, Kaggle submission files, and Kaggle API submission workflows.
---

# AutoDL GitHub Kaggle Experiment Workflow

## Purpose

This skill standardizes the workflow for H&M recommendation-system experiments where:

- Code is edited locally.
- Code is committed and pushed to GitHub.
- AutoDL is used for running experiments with large datasets.
- GitHub stores code, configs, scripts, README, and lightweight documentation.
- AutoDL stores full data, outputs, logs, checkpoints, submissions, and temporary experiment files.
- Kaggle API may be used to submit `submission.csv` files to the H&M competition.

The main project is usually:

```bash
/root/autodl-tmp/H-M-rec
```

The GitHub repository is usually:

```bash
https://github.com/cyy111dmn/H-M-rec.git
```

The Kaggle competition slug is usually:

```bash
h-and-m-personalized-fashion-recommendations
```

## Core Principles

1. Local machine is the source of truth for code.
2. GitHub is used for code synchronization.
3. AutoDL is used for data-heavy experiment execution.
4. Large datasets must stay on AutoDL and must not be committed to GitHub.
5. The `data/` directory must be protected.
6. The `kaggle.json` API token must never be printed, committed, or exposed.
7. Routine AutoDL updates should use `git pull origin main`.
8. Do not re-run `git init` every time.
9. Use single-variable experiments when possible.
10. Record experiment results with commit hashes.

## Important Safety Rules

Never suggest or execute these commands without explicitly warning the user:

```bash
rm -rf data
git reset --hard
git checkout -f
rm -rf .git
```

Before any destructive Git operation, always check:

```bash
pwd
ls
git status
ls data/
```

Never push these to GitHub:

```text
data/
outputs/
checkpoints/
logs/
submissions/
*.csv
*.parquet
*.pkl
*.pt
*.pth
*.ckpt
*.npy
*.npz
kaggle.json
.kaggle/
```

## Recommended .gitignore

The project root `.gitignore` should include:

```gitignore
# Data
data/
*.csv
*.parquet
*.pkl
*.npy
*.npz

# Model artifacts
checkpoints/
*.pt
*.pth
*.ckpt

# Experiment outputs
outputs/
logs/
submissions/

# Kaggle credentials
kaggle.json
.kaggle/

# Python cache
__pycache__/
*.pyc
.ipynb_checkpoints/

# System files
.DS_Store
```

If large files were accidentally staged, use:

```bash
git restore --staged data/
git restore --staged "*.csv"
git restore --staged "*.parquet"
git restore --staged "*.pkl"
git restore --staged "*.pt"
git restore --staged "*.pth"
git restore --staged "*.ckpt"
```

If large files were already committed but should remain locally, use:

```bash
git rm -r --cached data/
git rm --cached "*.csv"
git rm --cached "*.parquet"
git rm --cached "*.pkl"
git commit -m "remove large data files from git tracking"
git push origin main
```

## Standard Local Development Workflow

When the user has modified code locally, use this workflow:

```bash
git status
git diff
git add .
git diff --cached --name-only
git commit -m "describe the experiment or fix"
git push origin main
```

Before committing, check that large files are not staged:

```bash
git diff --cached --name-only
```

If `data/`, `.csv`, `.parquet`, `.pkl`, `.pt`, `.pth`, `.ckpt`, `kaggle.json`, or output files appear, stop and fix `.gitignore`.

## Standard AutoDL Workflow

For routine experiment execution on AutoDL:

```bash
cd /root/autodl-tmp/H-M-rec
pwd
git status
git pull origin main
ls data/
python main.py
```

If the experiment has a name or arguments:

```bash
python main.py --exp_name experiment_name
```

Always confirm the data files exist before running:

```bash
ls data/
```

Expected H&M files may include:

```text
articles.csv
customers.csv
transactions_train.csv
```

## First-Time AutoDL Repository Repair Workflow

Use this only when AutoDL local Git history is broken, or when the AutoDL project is not properly connected to GitHub.

This workflow protects the `data/` directory before rebuilding Git state.

```bash
cd /root/autodl-tmp/H-M-rec

pwd
ls
ls data/

mv data /root/autodl-tmp/data_tmp_HM

rm -rf .git
git init
git remote add origin https://github.com/cyy111dmn/H-M-rec.git
git fetch origin
git checkout -f origin/main -B main

mv /root/autodl-tmp/data_tmp_HM data

ls -la
ls data/
git log --oneline
```

If the GitHub default branch is `master` instead of `main`, inspect branches:

```bash
git branch -r
```

Then use:

```bash
git checkout -f origin/master -B master
```

This repair workflow should not be used as a daily workflow. For daily use, prefer:

```bash
git pull origin main
```

## If AutoDL Has Local Code Changes

Before pulling from GitHub:

```bash
git status
git diff
```

If the user wants to keep AutoDL edits:

```bash
git add .
git commit -m "save AutoDL local changes"
git pull origin main
```

If the user confirms AutoDL edits are useless and wants GitHub version only:

```bash
git fetch origin
git reset --hard origin/main
```

Warn that this discards uncommitted AutoDL code changes.

## If git pull Has Conflicts

Do not blindly overwrite.

First inspect:

```bash
git status
```

If the user wants to keep GitHub version only:

```bash
git fetch origin
git reset --hard origin/main
```

Warn that this discards local code changes.

If the user wants to preserve AutoDL changes before resolving conflicts:

```bash
mkdir -p /root/autodl-tmp/backup_autodl_changes
cp path/to/conflicted_file /root/autodl-tmp/backup_autodl_changes/
```

Then resolve conflicts manually.

## Experiment Naming Convention

Use clear experiment names:

```bash
python main.py --exp_name hot_recall_baseline
python main.py --exp_name itemcf_time_decay_v1
python main.py --exp_name itemcf_time_decay_v2
python main.py --exp_name item2vec_recall_v1
python main.py --exp_name two_tower_recall_v1
python main.py --exp_name recall_ranker_v1
```

Each experiment should record:

- experiment name
- Git commit hash
- changed module
- local validation MAP@12
- runtime
- memory usage
- Kaggle public LB score if submitted
- conclusion

Get current commit hash:

```bash
git rev-parse --short HEAD
```

## Recommended Experiment Log

Maintain a file named:

```bash
experiments.md
```

Suggested table:

```markdown
| Date | Commit | Experiment | Local MAP@12 | Public LB | Change | Conclusion |
|---|---|---|---:|---:|---|---|
| 2026-xx-xx | abc1234 | itemcf_time_decay_v2 | 0.xxxx | 0.xxxx | Added time decay to ItemCF | Improved recall diversity |
```

When creating a Kaggle submission, include the commit hash in the submission message.

## Kaggle API Setup on AutoDL

Use this section when the user wants to submit results to Kaggle.

If Kaggle CLI is not installed:

```bash
pip install kaggle
```

The user must create a Kaggle API token on the Kaggle website:

```text
Kaggle profile -> Account -> API -> Create New Token
```

This downloads:

```text
kaggle.json
```

On AutoDL, place it at:

```bash
~/.kaggle/kaggle.json
```

Then run:

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

Verify authentication:

```bash
kaggle competitions list
```

Never print the contents of `kaggle.json`.

Never commit `kaggle.json` to GitHub.

Make sure `.gitignore` includes:

```gitignore
kaggle.json
.kaggle/
```

## H&M Kaggle Submission Workflow

The H&M competition slug is usually:

```bash
h-and-m-personalized-fashion-recommendations
```

Before submitting, verify the submission file exists:

```bash
ls -lh submissions/submission.csv
head submissions/submission.csv
```

Expected H&M submission format:

```csv
customer_id,prediction
```

Each `prediction` row should contain exactly 12 space-separated article IDs.

Run this format check before submitting:

```bash
python - <<'PY'
import pandas as pd

path = "submissions/submission.csv"
df = pd.read_csv(path)

print(df.head())
print(df.columns.tolist())
print("rows:", len(df))

assert list(df.columns) == ["customer_id", "prediction"], "Columns must be customer_id,prediction"
assert df["customer_id"].notna().all(), "customer_id contains NA"
assert df["prediction"].notna().all(), "prediction contains NA"
assert df["prediction"].apply(lambda x: len(str(x).split())).eq(12).all(), "Each prediction must contain exactly 12 items"

print("submission format OK")
PY
```

Submit to Kaggle:

```bash
kaggle competitions submit \
  -c h-and-m-personalized-fashion-recommendations \
  -f submissions/submission.csv \
  -m "experiment message"
```

Recommended message format:

```bash
kaggle competitions submit \
  -c h-and-m-personalized-fashion-recommendations \
  -f submissions/submission.csv \
  -m "itemcf_time_decay_v2 commit=$(git rev-parse --short HEAD)"
```

After submission, check submissions:

```bash
kaggle competitions submissions \
  -c h-and-m-personalized-fashion-recommendations
```

Record:

- experiment name
- commit hash
- local validation MAP@12
- Kaggle public LB score
- submission file path
- submission message
- conclusion

## Full Daily Workflow

The normal full workflow is:

Local machine:

```bash
git status
git add .
git diff --cached --name-only
git commit -m "add experiment_name"
git push origin main
```

AutoDL:

```bash
cd /root/autodl-tmp/H-M-rec
git status
git pull origin main
ls data/
python main.py --exp_name experiment_name
```

Check generated submission:

```bash
ls -lh submissions/submission.csv
head submissions/submission.csv
```

Validate format:

```bash
python - <<'PY'
import pandas as pd

path = "submissions/submission.csv"
df = pd.read_csv(path)

assert list(df.columns) == ["customer_id", "prediction"], "Columns must be customer_id,prediction"
assert df["prediction"].notna().all(), "prediction contains NA"
assert df["prediction"].apply(lambda x: len(str(x).split())).eq(12).all(), "Each prediction must contain exactly 12 items"

print("submission format OK")
PY
```

Submit:

```bash
kaggle competitions submit \
  -c h-and-m-personalized-fashion-recommendations \
  -f submissions/submission.csv \
  -m "experiment_name commit=$(git rev-parse --short HEAD)"
```

Check submission:

```bash
kaggle competitions submissions \
  -c h-and-m-personalized-fashion-recommendations
```

## Result Download from AutoDL

If the user wants to download result files from AutoDL, suggest AutoDL file manager or `scp`.

Example:

```bash
scp -P PORT root@SERVER_IP:/root/autodl-tmp/H-M-rec/submissions/submission.csv .
```

Replace `PORT` and `SERVER_IP` with the actual AutoDL values.

For outputs:

```bash
scp -P PORT root@SERVER_IP:/root/autodl-tmp/H-M-rec/outputs/result.csv .
```

## Common User Explanation

If the user asks whether they should edit locally and run on AutoDL, explain:

Yes. The clean workflow is:

1. Edit code locally.
2. Commit and push to GitHub.
3. Start AutoDL.
4. Pull latest code on AutoDL.
5. Run experiments using AutoDL local data.
6. Generate `submission.csv`.
7. Check submission format.
8. Submit to Kaggle using Kaggle API.
9. Record the local score, public LB score, commit hash, and conclusion.

GitHub manages code. AutoDL manages large data and experiment outputs. Kaggle receives only verified submission files.

## What to Do When User Asks for Help

When the user asks for commands, first identify the situation:

- Local code needs to be pushed
- AutoDL needs to pull latest code
- AutoDL Git state is broken
- Data directory may be at risk
- Git conflict occurred
- Submission file needs checking
- Kaggle API needs setup
- Kaggle submission needs to be sent
- Experiment result needs recording

Then provide safe, copy-pasteable commands.

Do not give vague advice. Prefer exact commands.

## Final Safety Checklist

Before running an experiment:

```bash
pwd
git status
git rev-parse --short HEAD
ls data/
```

Before committing:

```bash
git diff --cached --name-only
```

Before Kaggle submission:

```bash
ls -lh submissions/submission.csv
head submissions/submission.csv
```

Before any destructive command:

```bash
pwd
ls
git status
ls data/
```

Never delete `data/` unless it has been explicitly backed up and the user has confirmed the action.
