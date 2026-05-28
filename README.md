# H&M 推荐系统 — 双阶段架构（多路召回 + LGBM 精排）

## 项目概述

H&M 时装推荐系统，基于 [Kaggle H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/) 竞赛数据。采用经典的 **召回 → 排序** 双阶段架构，在 CPU 环境下快速迭代验证，在 GPU 环境跑全量数据提交 Kaggle 打榜。

---

## 架构总览

```
                    ┌──────────────┐
                    │  137 万用户   │
                    │  3177 万交易  │
                    └──────┬───────┘
                           ▼
        ┌──────────────────────────────────┐
        │          召回层 (Recall)          │
        │  ┌─────────┐  ┌──────────┐       │
        │  │复购召回  │  │ItemCF召回│       │
        │  │(用户历史)│  │(协同过滤)│       │
        │  └────┬────┘  └────┬─────┘       │
        │       └────┬───────┘              │
        │       ┌────▼──────┐               │
        │       │ 热门兜底   │               │
        │       └────┬──────┘               │
        │            ▼                      │
        │   加权融合 → Top-100 候选          │
        └────────────────┬─────────────────┘
                         ▼
        ┌──────────────────────────────────┐
        │         排序层 (Ranker)           │
        │   LGBM Ranker (LambdaMART)       │
        │   特征: 价格/类目/颜色/年龄/...   │
        │        重排 → Top-12             │
        └────────────────┬─────────────────┘
                         ▼
                  ┌───────────┐
                  │ submission │
                  │   .csv    │
                  └───────────┘
```

---

## 评估指标

### MAP@12 (Mean Average Precision @ 12)

Kaggle 官方评估指标。对每个用户计算 AP@12，再对所有用户取平均。

**AP@12：**
```
        1    K  hits@k × rel@k
AP@12 = ─ ×  Σ  ────────────
       min(ground_truth, 12)  k=1     k

hits@k  = 第 k 个推荐是否命中
rel@k   = 前 k 个推荐中的命中数
```

**特点：**
- 同时考虑**命中率**和**排序质量**（命中的商品排得越靠前越好）
- 一个用户买得多件商品，只取前 12 件参与计算
- Kaggle Private Leaderboard 上 SOTA 约 **0.042-0.052**

### 辅助指标（线下调试用）

| 指标 | 含义 |
|------|------|
| Recall@K | 召回了多少比例的真实购买商品 |
| Popularity Coverage | 推荐结果中有多少比例来自热门商品 |
| 精排提升率 | (精排MAP - 召回MAP) / 召回MAP |

---

## 文件说明

### 核心模块 (src/)

| 文件 | 功能 |
|------|------|
| `recall_merged.py` | 多路召回管理器：复购召回 + ItemCF + 热门兜底，加权融合 |
| `features.py` | 特征工程：价格敏感度、类目偏好、颜色偏好、商品生命周期 |
| `ranker.py` | LGBM Ranker 训练器，支持 categorical 特征 |
| `data_loader.py` | 数据加载、按时间切分训练/验证集 |
| `metrics.py` | MAP@12 评估 |
| `sampling.py` | 用户采样工具 |
| `config.py` | 全局配置（路径、LGBM 参数） |

### 入口脚本

| 脚本 | 用途 | 数据规模 | 运行环境 |
|------|------|----------|----------|
| `simple_validation.py` | 快速验证召回+排序是否跑通 | 1000 人 | 本地 CPU |
| `validation_5w.py` | 5 万人级线下 AB 测试 | 5 万人 | 本地 CPU/GPU |
| `ablation_study.py` | 消融实验：逐项验证每个模块的贡献 | 5 万人 | 本地 CPU/GPU |
| `main.py` | 全量数据跑批，生成 Kaggle 提交文件 | 137 万人 | GPU 服务器 |
| `prepare_offline_data.py` | 按时间切分全量数据为离线训练/验证集 | 全量 | GPU 服务器 |

### 采样脚本

| 脚本 | 说明 |
|------|------|
| `create_full_sample.py` | 采样 10 万活跃用户 |
| `create_medium_sample.py` | 采样 1000 个最近一周有购买的用户 |
| `create_real_sample.py` | 采样 2 万活跃用户 (>=5 次购买) |

---

## 开发与提交工作流

```
 本地 (CPU, 小样本)              AutoDL (GPU, 全量)
 ─────────────────            ────────────────────
                          ┐
 1. simple_validation.py  │  快速验证改对了没有
    (1000用户, 1分钟)      │  报错立刻改
                          ┘
         ▼  通过
                          ┐
 2. validation_5w.py     │  线下 AB 测试
    (5万用户, ~5分钟)      │  确认 MAP 有提升
                          ┘
         ▼  通过
                          ┐
 3. ablation_study.py    │  消融实验
    (5万用户, ~10分钟)     │  确认每个模块都有效
                          ┘
         ▼  通过
                          ┐
 4. python main.py       │  全量跑批 → submission.csv
    (137万用户, GPU)       │  Kaggle 提交看分数
                          ┘
```

### 详细步骤

#### 第 1 步：本地快速验证（CPU，< 1 分钟）

```bash
# 确保 data/ 下有 transactions_train.csv 和 customers.csv
python prepare_offline_data.py        # 按时间切分，生成 offline_data/
python simple_validation.py           # 热门Baseline + 多路召回 + LGBM精排对比
```

**验证内容：** 代码能否跑通、feature 是否正常、MAP 是否有基本值。

#### 第 2 步：线下 AB 测试（CPU，~5 分钟）

```bash
python validation_5w.py               # 5 万人级，完整召回→排序→评估
```

**验证内容：** 在更大规模上确认 MAP 提升，同时看内存和耗时是否可接受。

#### 第 3 步：消融实验（CPU/GPU，~10 分钟）

```bash
python ablation_study.py              # 7 组对比实验
```

输出 ablation_results.csv，包含：
- baseline / recall(full) / ranker(full) MAP@12
- recall(no_repurchase) / recall(no_itemcf)
- ranker(no_price) / ranker(no_preference_match)

**验证内容：** 每个模块的边际贡献，决定哪些改动能真正涨分。

#### 第 4 步：全量跑批 → 提交 Kaggle（GPU，~30-60 分钟）

在 AutoDL 上：

```bash
# 确保 data/ 下有全量数据
python main.py                        # 跑全量 137 万用户
# 输出 submission.csv
```

将 `submission.csv` 提交到 [Kaggle Competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/)。

---

## 快速上手

### 安装依赖

```bash
pip install pandas numpy lightgbm scipy psutil tqdm pyarrow
```

### 数据准备

从 Kaggle 竞赛页面下载数据放在 `data/` 目录下：
- `data/transactions_train.csv`（~3.5 GB）
- `data/customers.csv`
- `data/articles.csv`

### 运行示例

```bash
# 1. 准备离线验证数据
python prepare_offline_data.py

# 2. 快速验证（1000 人）
python simple_validation.py

# 3. 5 万人验证
python validation_5w.py

# 4. 消融实验
python ablation_study.py

# 5. 全量跑批（GPU 服务器）
python main.py
```

---

## 注意事项

- `.gitignore` 已配置：`data/`、`__pycache__/`、`.autodl/` 不会被提交
- 所有 ID 在 `recall_merged.py` 中被统一为 `str` 类型，避免类型不匹配
- 精排训练样本构造时会将 ID 转为 `str` 再匹配，这是已知的性能瓶颈，后续优化
