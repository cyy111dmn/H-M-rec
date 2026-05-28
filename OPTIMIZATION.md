# H&M 推荐系统 — 优化清单

按「预估收益 × 改动难度」排列，优先级从高到低。

---

## P0 — Bug 修复

### 1. RecallManager 融合权重是拍脑袋的

**位置：** `src/recall_merged.py:46-52`

当前权重是手动调的：
```python
repurchase_weight=3.0, itemcf_weight=1.5, popularity_weight=0.8
```

没有依据，也没有验证过。应该用 ablation_study.py 跑网格搜索来确认最优组合。

**改动量：** 在 `ablation_study.py` 中加几组权重组合对比即可。
**预估收益：** MAP ↑0-2%

---

### 2. LGBM 参数不一致

**位置：** `src/config.py:16-22` vs `src/ranker.py:47-59`

- `config.py` 用 `boosting_type='dart'`, `metric='ndcg'`, `n_estimators=100`
- `ranker.py` 用 `metric='map'`, `learning_rate=0.05`, `n_estimators=300`, `boosting_type='gbdt'`

**问题：**
1. `dart` 比 `gbdt` 慢 3-5 倍，对推荐场景没什么必要
2. `config.py` 的参数和 `ranker.py` 的默认参数是两套，看调用方用了哪套
3. `metric='ndcg'` 但线下评估用 `MAP`，优化目标和评估指标不一致

**建议：** 统一到 `ranker.py` 的默认参数，删除 `config.py` 中过时的 `LGBM_PARAMS`。
**预估收益：** 训练速度提升 3-5 倍 + MAP ↑0-1%

---

### 3. `_normalize_items` 兜底逻辑可能吞掉错误类型

**位置：** `src/metrics.py:30`

```python
return [x]  # 兜底：如果类型不在上面任何一种，包成列表
```

如果传进来一个字符串 `"12345"`，会变成 `["12345"]` 而不是 `[12345]`，导致推荐 ID 和标签 ID 类型不一致而永远匹配不上，MAP 直接变 0。

**建议：** 兜底分支里加一个 `warnings.warn` 或 `print`，方便排查。
**预估收益：** 防坑。

---

## P1 — 召回层优化

### 4. PopularityRecall 只用最后一周算热门

**位置：** `src/recall_merged.py:281-284`

```python
last_week = self.train_trans["week"].min()
last_week_data = self.train_trans[self.train_trans["week"] == last_week]
```

最后一周可能有异常（比如某天爆款断货），应该用多周指数衰减加权：

```
score(item) = Σ exp(-α × week_diff) × count_in_week
```

**建议：** 把 `_get_popular_items` 改成跨多周加权聚合，α 可配置。
**改动量：** 约 10 行代码。
**预估收益：** MAP ↑0.5-1%

---

### 5. ItemCF 的 Jaccard 没做热门商品降权

**位置：** `src/recall_merged.py:381-395`

当前 Jaccard 相似度：`co_occur / (count(A) + count(B) - co_occur)`

热门商品（比如白色基本款 T 恤）会跟几乎所有商品共现，Jaccard 被高估。应该加 IDF 风格的降权：

```
sim(i,j) = co_occur(i,j) / sqrt(count(i) × count(j))     # 余弦归一化
```

或者加平滑因子：

```
sim(i,j) = co_occur(i,j) / (count(i) + count(j) - co_occur(i,j) + λ)
```

**建议：** 在 `_prepare_matrices` 中增加一个 `normalize` 参数可选余弦/ Jaccard。
**预估收益：** MAP ↑0.5-1.5%

---

### 6. RepurchaseRecall 是 Python 循环

**位置：** `src/recall_merged.py:304-311`

```python
for user_id, group in sorted_trans.groupby("customer_id"):
    unique_items = []
    for item_id in group["article_id"]:
        ...
```

对于 137 万用户，这个 for 循环会是瓶颈。可以改写为：

```python
user_purchases = (
    sorted_trans.groupby("customer_id")["article_id"]
    .agg(lambda x: list(dict.fromkeys(x)))
    .to_dict()
)
```

`dict.fromkeys` 保持顺序且去重，是纯 C 代码。

**预估收益：** 复购召回速度 ↑5-10 倍（全量数据从几十秒降到几秒）。

---

### 7. PopularityRecall 所有用户共享同一个列表对象

**位置：** `src/recall_merged.py:287`

```python
return {user_id: self.popular_items for user_id in val_users}
```

所有 value 指向同一个列表对象，如果有人修改了其中一个，全部受影响。应该 `self.popular_items[:]` 浅拷贝。

**预估收益：** 防坑，几乎无性能损失。

---

## P2 — 排序层优化

### 8. 精排样本构造的 ID 类型转换效率低

**位置：** `ablation_study.py`、`validation_5w.py`、`simple_validation.py`

每个脚本都在循环中做：
```python
str_source_info = {str(u): {str(i): v ...} ...}
str_ground_truth = {str(k): set([str(x) for x in v]) ...}
```

这几百万次 `str()` 转换在 CPU 上可以跑几十秒。应该在 `RecallManager._preprocess_data()` 中就把所有 ID 统一类型，后续不再转换。

**改动量：** 集中到 `recall_merged.py` 一层处理，删除各脚本的重复转换。
**预估收益：** 精排样本构造速度 ↑30-50% (全量数据可省几分钟)。

---

### 9. 精排特征缺少用户价格分位数

**位置：** `src/features.py:91-102`

当前只有 `user_avg_price`（均值），均值对离群值敏感。应该加：

- `user_price_p50` — 用户历史购买价格中位数
- `user_price_p90` — 用户能接受的价格上限
- `user_price_p10` — 用户买过的最低价

**建议：** 用 `groupby + quantile()` 一次性算多个分位数。
**预估收益：** MAP ↑0.3-1%

---

### 10. 缺少商品新旧程度的交叉特征

**位置：** `src/features.py:104-118`

`item_age_weeks` 已计算，但没有和 `is_repurchase` 做交叉特征。

新上架的商品 + 用户复购 → 可能是定期补货（比如每季买一次基本款），这类组合预测力更强。

**建议：** 加特征 `is_new_repurchase = (item_age_weeks < threshold) & (is_repurchase == 1)`
**预估收益：** MAP ↑0.2-0.5%

---

### 11. 缺少用户活跃度分桶特征

当前 `user_activity` 是原始购买次数（从 1 到数千），树模型对这种长尾分布不敏感。应该做对数变换或分桶：

- `user_activity_log = log(user_activity + 1)`
- `user_activity_bucket = pd.qcut(user_activity, q=5)` — 按分位数分成 5 档

**预估收益：** MAP ↑0.2-0.5%

---

## P3 — 工程化

### 12. 爬取 tools 函数重复了 5 份

`get_memory_usage()` / `print_memory_usage()` 出现在 6 个文件中：
- `src/recall_merged.py`
- `simple_validation.py`
- `validation_5w.py`
- `ablation_study.py`
- `main.py`
- `prepare_offline_data.py`

`reduce_mem_usage()` 出现在 `validation_5w.py` 和 `ablation_study.py` 中。

**建议：** 提取到 `src/utils.py`

---

### 13. `create_ranking_samples` 重复了 3 份

`simple_validation.py`、`validation_5w.py`、`ablation_study.py` 各有一份几乎一样的实现。

**建议：** 提取到 `src/features.py` 或新建 `src/ranking.py`

---

### 14. 没有 requirements.txt

需要用户自己猜依赖。写一份：

```
pandas>=1.3.0
numpy>=1.21.0
lightgbm>=3.3.0
scipy>=1.7.0
psutil>=5.8.0
tqdm>=4.62.0
pyarrow>=6.0.0
```

---

### 15. 全量 main.py 的分批预测没有错误恢复

如果在第 12 万个用户处 crash，前面写了一半的 submission.csv 就废了，得重头跑。

**建议：** 每批写完 flush 到磁盘，重启时检测已有结果并跳过。
**预估收益：** 省心，137 万用户跑 1 小时挂了不白费。

---

## P4 — 远期/探索性优化

### 16. 召回融合权重网格搜索

`RecallManager` 的 6 个权重参数（3 个来源权重 + 3 个 rank 权重）应该用网格搜索来调。可以在 `ablation_study.py` 里加一个搜索模式，遍历几十组组合。

### 17. ItemCF 特征扩展

当前 ItemCF 只用了购买行为。H&M 有商品详情（颜色、品类、材质），可以做基于内容的相似度，与协同过滤分数做线性融合。

### 18. 增加 UserCF 召回

当前只有 ItemCF 和复购。可以加一路 UserCF（找相似用户买的商品），特别是对新商品的覆盖有帮助。

### 19. 精排 Label 加权

当前所有 `purchased=1` 的样本权重相同。在验证周第 1 天买的比第 7 天买的更难预测（特征更旧），可以按时间设置样本权重。

### 20. Stacking / 模型融合

当前只有 LGBM 一路精排。可以加一路简单的 LR 或 FM 做第二层融合。

---

## 优先级速查表

| # | 优化项 | 预估 MAP 提升 | 难度 | 推荐顺序 |
|---|--------|-------------|------|---------|
| 1 | 融合权重网格搜索 | ↑0-2% | 简单 | ① |
| 2 | LGBM 参数统一 | ↑0-1% + 3-5x 加速 | 简单 | ② |
| 4 | 多周指数热门 | ↑0.5-1% | 简单 | ③ |
| 5 | ItemCF IDF 降权 | ↑0.5-1.5% | 中等 | ④ |
| 8 | 精排样本 ID 转换优化 | 速度 ↑30-50% | 中等 | ⑤ |
| 9 | 价格分位数特征 | ↑0.3-1% | 简单 | ⑥ |
| 12-14 | 提取 utils / 重复代码清理 | 可维护性 | 简单 | 随时做 |
| 15 | 全量跑批错误恢复 | 防 crash 白跑 | 中等 | 上线前做 |
