#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速本地验证脚本（轻量级）
从 offline_data parquet 采样少量用户，跑热门Baseline + 多路召回，验证代码无报错。
"""

import pandas as pd
import numpy as np
import os
import sys
import time

sys.path.append('src')

from src.metrics import calculate_map_at_k


def quick_load():
    print("📂 加载 offline_data parquet（采样）...", flush=True)
    offline_dir = './offline_data'

    train_trans = pd.read_parquet(f'{offline_dir}/offline_train_trans.parquet')
    val_customers = pd.read_parquet(f'{offline_dir}/offline_users.parquet')
    val_ground_truth = pd.read_parquet(f'{offline_dir}/offline_val_truth.parquet')

    if 'purchased_articles' not in val_ground_truth.columns:
        if 'article_id' in val_ground_truth.columns:
            if not isinstance(val_ground_truth['article_id'].iloc[0], (list, np.ndarray)):
                val_ground_truth = val_ground_truth.groupby('customer_id')['article_id'].apply(list).reset_index()
            val_ground_truth = val_ground_truth.rename(columns={'article_id': 'purchased_articles'})

    return train_trans, val_customers, val_ground_truth


def run_popularity_baseline(train_trans, val_customers, val_ground_truth):
    print("\n" + "=" * 60 + "\n🔥 热门Baseline\n" + "=" * 60)
    train_trans = train_trans.copy()
    train_trans['t_dat'] = pd.to_datetime(train_trans['t_dat'])
    max_date = train_trans['t_dat'].max()
    train_trans['week'] = (max_date - train_trans['t_dat']).dt.days // 7

    last_week_data = train_trans[train_trans['week'] == train_trans['week'].min()]
    popular_items = last_week_data.groupby('article_id').size().sort_values(ascending=False).head(50).index.tolist()

    truth_dict = val_ground_truth.set_index('customer_id')['purchased_articles'].to_dict()
    predicted_dict = {uid: popular_items[:12] for uid in truth_dict}
    map_score = calculate_map_at_k(truth_dict, predicted_dict, k=12)
    print(f"🎯 热门Baseline MAP@12: {map_score:.6f}")
    return map_score


def run_multi_recall(train_trans, val_customers, val_ground_truth):
    print("\n" + "=" * 60 + "\n🎯 多路召回\n" + "=" * 60)
    from src.recall_merged import RecallManager

    mgr = RecallManager(
        train_trans=train_trans,
        val_customers=val_customers,
        top_n=50,
        itemcf_top_k=50,
        max_items=5000,
        recall_cutoff=50,
    )
    final_recs, source_info = mgr.multi_recall_with_source_info()

    truth_dict = val_ground_truth.set_index('customer_id')['purchased_articles'].to_dict()
    map_score = calculate_map_at_k(truth_dict, final_recs, k=12)
    print(f"🎯 多路召回 MAP@12: {map_score:.6f}")
    return map_score


def main():
    print("🚀 快速本地验证 — 热门Baseline + 多路召回")

    # 1. 加载数据
    train_trans, val_customers, val_ground_truth = quick_load()

    # 2. 采样少量用户加速
    all_users = val_customers['customer_id'].unique()
    sample_size = min(2000, len(all_users))
    np.random.seed(42)
    sampled = set(np.random.choice(all_users, sample_size, replace=False))
    val_customers_small = val_customers[val_customers['customer_id'].isin(sampled)].copy()
    val_truth_small = val_ground_truth[val_ground_truth['customer_id'].isin(sampled)].copy()
    print(f"👥 采样 {sample_size} 用户用于验证")

    # 3. 热门Baseline
    baseline_map = run_popularity_baseline(train_trans, val_customers_small, val_truth_small)

    # 4. 多路召回
    recall_map = run_multi_recall(train_trans, val_customers_small, val_truth_small)

    print("\n" + "=" * 60)
    print(f"📊 汇总 — 采样 {sample_size} 用户")
    print(f"   热门Baseline MAP@12: {baseline_map:.6f}")
    print(f"   多路召回     MAP@12: {recall_map:.6f}")
    print(f"   召回相对提升:        {(recall_map - baseline_map) / (baseline_map + 1e-9) * 100:+.2f}%")
    print("✅ 本地验证通过！可以推送代码到 GitHub 并上 AutoDL 跑全量。")
    print("=" * 60)


if __name__ == "__main__":
    main()
