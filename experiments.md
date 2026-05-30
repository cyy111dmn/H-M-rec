# 实验记录

每次改动的实验结果追踪。

## 格式

| 日期 | Commit | 改动 | 5w MAP@12 | 全量 | Kaggle LB | 结论 |
|---|---|---|---:|---:|---:|---|
| yyyy-MM-dd | abcdef | 改了什么 | 0.xxxxx | y/n | 0.xxxxx | 结论 |

---

## 实验列表

| 日期 | Commit | 改动 | 5w MAP@12 | 全量 | Kaggle LB | 结论 |
|---|---|---|---:|---:|---:|---|
| 2026-05-28 | 826b854 | 初始提交：多路召回（热门+复购+ItemCF Jaccard），纯召回出提交文件 | — | ✅ | **0.014** | 基线 |
| 2026-05-28 | e9a10a3 | 添加 SSH 一键实验工作流 + AutoDL 部署 | — | — | — | 工程化 |
| 2026-05-29 | 9042a98 | simple_validation.py 改为本地离线 parquet 验证 | 0.015109 (2k) | — | — | 本地验证可用 |
| 2026-05-29 | 9b84eb5 | 融合权重网格搜索 + RecallManager.grid_search_weights | *见下方* | — | — | 代码就绪 |
| 2026-05-29 | 0e96bbb | 提取 utils.py / 多周热门 / 余弦归一化 / Repurchase加速 / config清理 | 0.012660 ❌ | — | — | 召回算法改动导致 MAP 下降 |
| 2026-05-29 | f155a22 | 回滚多周热门+余弦；添加 DeepFM 模型 + 训练脚本 + main.py --use_deepfm | **0.017719** (LGBM) | — | — | 回滚后召回恢复，LGBM 精排有效 |
| 2026-05-29 | 8e7d885 | 精排特征工程：价格分位数(p50/p10/p90) + 活跃度对数/分桶 + 新品复购交叉 | *与网格搜索合并* | — | — | 特征扩展 |
| 2026-05-29 | baa6be2 | **网格搜索最佳权重**: repurchase=2.0, itemcf=1.5, popularity=0.8 | **0.017785** (+9.2%) | ✅ | **0.01604** (+14.6%) | 最佳权重提升有效，线上 +14.6% |
| 2026-05-29 | f428276 | 添加 experiments.md 实验记录文件 | — | — | — | 工程化 |
| 2026-05-29 | 8ef77a5 | 添加 main.py --use_ranker（LGBM 精排全量推理）；跑 DeepFM 训练 | 0.012867 (DeepFM) ❌ | — | — | DeepFM MAP 低于纯召回，需调参 |
| 2026-05-30 | c53332d | main.py 分批预测 + 预计算特征 + LGBM 全量跑批 | — | ✅ | **0.008** ❌ | LGBM 精排全量导致 MAP 大跌，训练标签穿越过拟合。回退纯召回 |

## 网格搜索明细

权重组合 (repurchase / itemcf / popularity) 与 MAP@12 的关系：

| repurchase | itemcf | popularity | MAP@12 |
|:----------:|:-----:|:----------:|:------:|
| **2.0** | **1.5** | **0.8** | **0.017785** 🥇 |
| **2.5** | **2.0** | **0.8** | **0.017785** 🥇 |
| **3.0** | **2.5** | **0.8** | **0.017785** 🥇 |
| 3.5 | 3.0 | 0.8 | 0.017785 🥇 |
| 2.5 | 2.5 | 0.8 | 0.017704 |
| 2.0 | 2.0 | 0.8 | 0.017704 |
| 3.0 | 3.0 | 0.5 | 0.017442 |
| — | — | — | — |
| 3.0 (默认) | 1.5 (默认) | 0.8 (默认) | 0.016289 |

> 关键发现：**popularity_weight=0.8 固定**时，repurchase / itemcf ≈ 4/3 比例最优。

## 全量提交流程

```bash
# 1. 确认生成文件
ls -lh submissions/submission_*.csv
head submissions/submission_*.csv

# 2. 压缩
tar czf submission_$(git rev-parse --short HEAD).tar.gz -C submissions/ .

# 3. 下载（AutoDL 网页控制台）
# 或 scp -P PORT root@SERVER_IP:/root/autodl-tmp/H-M-rec/submissions/submission_*.csv .

# 4. 提交 Kaggle
kaggle competitions submit \
  -c h-and-m-personalized-fashion-recommendations \
  -f submissions/submission_grid_search_best.csv \
  -m "grid_search_best commit=baa6be2 weights=(2.0,1.5,0.8)"
```
