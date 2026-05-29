#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中大规模(5万级) 召回与排序验证脚本
核心修复：
1. baseline / recall / ranker 全都在同一批 valid_users 上评估
2. ranker 训练与推理口径统一
3. 防止“训练即评估”的线下穿越
"""

import gc
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('src')

from src.features import extract_advanced_features_for_twostage
from src.metrics import calculate_map_at_k
from src.ranker import train_lgbm_ranker
from src.utils import print_memory_usage, reduce_mem_usage


def load_parquet_data():
    print("📂 加载预切分好的离线 Parquet 数据...", flush=True)
    offline_dir = './offline_data'
    raw_dir = './data'

    try:
        train_trans = pd.read_parquet(f'{offline_dir}/offline_train_trans.parquet')
        val_customers = pd.read_parquet(f'{offline_dir}/offline_users.parquet')
        val_ground_truth = pd.read_parquet(f'{offline_dir}/offline_val_truth.parquet')

        if 'purchased_articles' not in val_ground_truth.columns:
            if 'article_id' in val_ground_truth.columns:
                if not isinstance(val_ground_truth['article_id'].iloc[0], (list, np.ndarray)):
                    print("⚙️ 正在将真实购买流水聚合成用户列表格式...")
                    val_ground_truth = val_ground_truth.groupby('customer_id')['article_id'].apply(list).reset_index()
                val_ground_truth = val_ground_truth.rename(columns={'article_id': 'purchased_articles'})

        customers = pd.read_csv(f'{raw_dir}/customers.csv')
        articles = pd.read_csv(f'{raw_dir}/articles.csv')

        valid_users = val_customers['customer_id'].unique()
        customers = customers[customers['customer_id'].isin(valid_users)].copy()

        print("✅ 数据与画像加载完成!")
        return train_trans, val_customers, val_ground_truth, customers, articles
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None, None, None, None


def create_ranking_samples(val_customers, val_ground_truth, recall_recs, source_info=None):
    print("\n📋 构造精排样本表...", flush=True)

    str_source_info = {}
    if source_info:
        for u, items_dict in source_info.items():
            str_source_info[str(u)] = {str(i): v for i, v in items_dict.items()}

    ground_truth = val_ground_truth.set_index('customer_id')['purchased_articles'].to_dict()
    str_ground_truth = {str(k): set([str(x) for x in v]) for k, v in ground_truth.items()}

    samples = []
    for user_id in tqdm(val_customers['customer_id'], desc="生成精排正负样本"):
        if user_id not in recall_recs:
            continue

        safe_u = str(user_id)
        purchased_items = str_ground_truth.get(safe_u, set())

        for item_id in recall_recs[user_id]:
            safe_i = str(item_id)
            sample = {
                'customer_id': user_id,
                'article_id': item_id,
                'purchased': 1 if safe_i in purchased_items else 0
            }

            if source_info and safe_u in str_source_info and safe_i in str_source_info[safe_u]:
                source = str_source_info[safe_u][safe_i]
                sample.update({
                    'is_from_repurchase': source.get('is_from_repurchase', 0),
                    'is_from_itemcf': source.get('is_from_itemcf', 0),
                    'is_from_popularity': source.get('is_from_popularity', 0),
                    'repurchase_rank': source.get('repurchase_rank', 999),
                    'itemcf_rank': source.get('itemcf_rank', 999)
                })
            else:
                sample.update({
                    'is_from_repurchase': 0,
                    'is_from_itemcf': 0,
                    'is_from_popularity': 0,
                    'repurchase_rank': 999,
                    'itemcf_rank': 999
                })

            samples.append(sample)

    train_df = pd.DataFrame(samples)
    train_df = reduce_mem_usage(train_df)
    return train_df


def subset_truth_dict(val_ground_truth, target_users):
    sub_truth = val_ground_truth[val_ground_truth['customer_id'].isin(target_users)].copy()
    return sub_truth.set_index('customer_id')['purchased_articles'].to_dict()


def run_popularity_baseline(train_trans, eval_users, eval_truth_dict):
    print("\n" + "=" * 60 + "\n🔥 热门Baseline实验\n" + "=" * 60)

    train_trans = train_trans.copy()
    train_trans['t_dat'] = pd.to_datetime(train_trans['t_dat'])
    max_date = train_trans['t_dat'].max()
    train_trans['week'] = (max_date - train_trans['t_dat']).dt.days // 7

    last_week_data = train_trans[train_trans['week'] == train_trans['week'].min()]
    popular_items = last_week_data.groupby('article_id').size().sort_values(ascending=False).head(50).index.tolist()

    predicted_dict = {user_id: popular_items[:12] for user_id in eval_users}
    map_score = calculate_map_at_k(eval_truth_dict, predicted_dict, k=12)
    print(f"🎯 热门Baseline MAP@12: {map_score:.6f}")
    return {'map_at_k': map_score}


def run_multi_recall_with_source(train_trans, eval_users_df, eval_truth_dict):
    print("\n" + "=" * 60 + "\n🎯 多路召回实验（带来源信息）\n" + "=" * 60)
    from src.recall_merged import RecallManager

    recall_manager = RecallManager(
        train_trans=train_trans,
        val_customers=eval_users_df,
        top_n=150,
        itemcf_top_k=150,
        max_items=5000,
        recall_cutoff=100
    )
    final_recs, source_info = recall_manager.multi_recall_with_source_info()

    map_score = calculate_map_at_k(eval_truth_dict, final_recs, k=12)
    print(f"🎯 多路召回 MAP@12: {map_score:.6f}")
    return {'map_at_k': map_score}, final_recs, source_info


def run_lgbm_ranking_experiment(
    train_trans,
    train_users_df,
    valid_users_df,
    valid_truth_dict,
    recall_recs_train,
    source_info_train,
    recall_recs_valid,
    source_info_valid,
    customers,
    articles
):
    print("\n" + "=" * 60 + "\n🎯 LGBM 精排实验\n" + "=" * 60)

    rank_train_df = create_ranking_samples(train_users_df, valid_truth_full_df_for_train_stub(train_users_df), recall_recs_train, source_info_train)
    rank_valid_df = create_ranking_samples(valid_users_df, valid_truth_full_df_from_dict(valid_truth_dict), recall_recs_valid, source_info_valid)

    full_feature_df = pd.concat([rank_train_df, rank_valid_df], axis=0, ignore_index=True)
    full_feature_df = extract_advanced_features_for_twostage(full_feature_df, train_trans, customers, articles)
    full_feature_df = reduce_mem_usage(full_feature_df)

    rank_train_df = full_feature_df[full_feature_df['customer_id'].isin(train_users_df['customer_id'])].copy()
    rank_valid_df = full_feature_df[full_feature_df['customer_id'].isin(valid_users_df['customer_id'])].copy()

    del full_feature_df
    gc.collect()

    feature_cols = [
        'item_popularity', 'user_activity', 'is_repurchase',
        'is_from_repurchase', 'is_from_itemcf', 'is_from_popularity',
        'repurchase_rank', 'itemcf_rank',
        'is_category_matched', 'is_color_matched',
        'item_price', 'user_avg_price', 'price_diff', 'item_age_weeks',
        'age', 'club_member_status', 'fashion_news_frequency',
        'product_group_name', 'index_group_name', 'colour_group_name', 'graphical_appearance_name'
    ]

    cat_columns = [
        'club_member_status', 'fashion_news_frequency',
        'product_group_name', 'index_group_name',
        'colour_group_name', 'graphical_appearance_name'
    ]

    print("🚀 训练 LGBM 模型...", flush=True)
    lgbm_params = {
        'objective': 'lambdarank',
        'metric': 'map',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'n_estimators': 300,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    ranker = train_lgbm_ranker(
        rank_train_df,
        feature_cols,
        lgbm_params,
        categorical_cols=cat_columns
    )

    print("🎯 在同一批 valid_users 上做精排评估...")
    batch_size = 500000
    preds = []
    for i in range(0, len(rank_valid_df), batch_size):
        preds.append(ranker.predict(rank_valid_df[feature_cols].iloc[i:i + batch_size]))
    rank_valid_df['score'] = np.concatenate(preds)

    sorted_df = rank_valid_df[['customer_id', 'article_id', 'score']].sort_values(
        ['customer_id', 'score'],
        ascending=[True, False]
    )
    top_items_df = sorted_df.groupby('customer_id').head(12)
    ranked_recs = top_items_df.groupby('customer_id')['article_id'].apply(list).to_dict()

    for user_id in valid_users_df['customer_id']:
        if user_id not in ranked_recs:
            ranked_recs[user_id] = []

    map_score = calculate_map_at_k(valid_truth_dict, ranked_recs, k=12)
    print(f"🎯 LGBM 精排 MAP@12: {map_score:.6f}")

    del rank_train_df, rank_valid_df
    gc.collect()

    return {'map_at_k': map_score}


def valid_truth_full_df_from_dict(truth_dict):
    return pd.DataFrame({
        'customer_id': list(truth_dict.keys()),
        'purchased_articles': list(truth_dict.values())
    })


def valid_truth_full_df_for_train_stub(train_users_df):
    # 训练样本打标签仍来自验证周真实标签，但只用于 train_users 部分
    # 这里会在 main 里被替换成真实 train_truth_df
    raise RuntimeError("请勿直接调用该函数")


def main():
    print("🚀 启动 5万人级 召回->排序 全链路压测")

    train_trans, val_customers, val_ground_truth, customers, articles = load_parquet_data()
    if train_trans is None:
        return

  
    all_users = val_customers['customer_id'].astype(str).to_numpy(copy=True)
    np.random.seed(42)
    np.random.shuffle(all_users)

    split_idx = int(len(all_users) * 0.8)
    train_users = set(all_users[:split_idx])
    valid_users = set(all_users[split_idx:])

    print(f"🔪 用户切分完成: train_users={len(train_users)}, valid_users={len(valid_users)}")

    train_users_df = val_customers[val_customers['customer_id'].isin(train_users)].copy()
    valid_users_df = val_customers[val_customers['customer_id'].isin(valid_users)].copy()

    train_truth_df = val_ground_truth[val_ground_truth['customer_id'].isin(train_users)].copy()
    valid_truth_df = val_ground_truth[val_ground_truth['customer_id'].isin(valid_users)].copy()

    train_truth_dict = train_truth_df.set_index('customer_id')['purchased_articles'].to_dict()
    valid_truth_dict = valid_truth_df.set_index('customer_id')['purchased_articles'].to_dict()

    global valid_truth_full_df_for_train_stub
    valid_truth_full_df_for_train_stub = lambda _: train_truth_df

    baseline_results = run_popularity_baseline(train_trans, list(valid_users), valid_truth_dict)

    multi_recall_train_results, multi_recs_train, source_info_train = run_multi_recall_with_source(
        train_trans, train_users_df, train_truth_dict
    )
    multi_recall_valid_results, multi_recs_valid, source_info_valid = run_multi_recall_with_source(
        train_trans, valid_users_df, valid_truth_dict
    )

    lgbm_results = run_lgbm_ranking_experiment(
        train_trans,
        train_users_df,
        valid_users_df,
        valid_truth_dict,
        multi_recs_train,
        source_info_train,
        multi_recs_valid,
        source_info_valid,
        customers,
        articles
    )

    print("\n" + "★" * 60 + "\n🏆 5W人线下验证真实战报\n" + "★" * 60)
    print(f"热门Baseline (valid_users MAP@12):  {baseline_results['map_at_k']:.6f}")
    print(f"多路召回 (valid_users MAP@12):      {multi_recall_valid_results['map_at_k']:.6f}")
    print(f"多维精排 (valid_users MAP@12):      {lgbm_results['map_at_k']:.6f}")

    improvement = (
        (lgbm_results['map_at_k'] - multi_recall_valid_results['map_at_k']) /
        (multi_recall_valid_results['map_at_k'] + 1e-9) * 100
    )
    print(f"📈 精排层相对召回提升:                {improvement:+.2f}%")


if __name__ == "__main__":
    main()