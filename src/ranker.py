#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Iterable, List, Optional

import lightgbm as lgb
import pandas as pd


def train_lgbm_ranker(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    params: dict,
    group_col: str = 'customer_id',
    label_col: str = 'purchased',
    categorical_cols: Optional[Iterable[str]] = None,
):
    """训练 LightGBM Ranker。

    关键修复：
    1. 强制按 group_col 排序后再构造 group。
    2. 自动识别 pandas category 特征并传给 LightGBM。
    3. 默认使用 MAP 常见配置，避免参数缺失导致训练口径混乱。
    """
    required_cols = set(feature_cols) | {group_col, label_col}
    missing_cols = [c for c in required_cols if c not in train_df.columns]
    if missing_cols:
        raise ValueError(f"train_df 缺少必要列: {missing_cols}")

    print(f"对数据按 {group_col} 排序并构造 group...", flush=True)
    df = train_df.sort_values(group_col).reset_index(drop=True).copy()
    group_sizes = df.groupby(group_col, sort=False).size().tolist()

    X_train = df[feature_cols]
    y_train = df[label_col].astype(int)

    if categorical_cols is None:
        categorical_cols = [
            col for col in feature_cols
            if col in X_train.columns and str(X_train[col].dtype) == 'category'
        ]
    else:
        categorical_cols = [col for col in categorical_cols if col in feature_cols]

    default_params = {
        'objective': 'lambdarank',
        'metric': 'map',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_estimators': 300,
        'n_jobs': -1,
        'verbose': -1,
    }
    final_params = {**default_params, **(params or {})}

    print('开始训练 LGBMRanker...', flush=True)
    ranker = lgb.LGBMRanker(**final_params)

    fit_kwargs = {
        'group': group_sizes,
    }
    if categorical_cols:
        fit_kwargs['categorical_feature'] = list(categorical_cols)

    ranker.fit(X_train, y_train, **fit_kwargs)
    return ranker