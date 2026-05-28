#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量跑批引擎 - 双阶段架构（多路召回 + LGBM精排）
修复点：
1. RecallManager 强制关键字参数
2. 训练与推理统一使用 train_lgbm_ranker
3. 训练与提交特征列保持一致
"""

import gc
import os
import sys

import pandas as pd
import psutil
from tqdm import tqdm

sys.path.append('src')

from src.data_loader import load_id_mapping, load_parquet_data, prepare_validation_data, split_data_by_time
from src.features import extract_advanced_features_for_twostage
from src.ranker import train_lgbm_ranker
from src.recall_merged import RecallManager


def print_memory_usage(step):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"[内存监控] {step}: {mem:.2f} MB", flush=True)


def load_full_data():
    print("📂 加载全量数据...", flush=True)
    data_dir = './data'
    raw_customers = pd.read_csv(f'{data_dir}/customers.csv')
    raw_articles = pd.read_csv(f'{data_dir}/articles.csv')
    transactions, customers = load_parquet_data(data_dir)
    uint_to_hex_cust = load_id_mapping(data_dir)
    return transactions, customers, uint_to_hex_cust, raw_customers, raw_articles


def create_ranking_samples_full(val_customers, val_ground_truth, recall_recs, source_info):
    print("📋 构造精排训练样本表...", flush=True)

    str_source_info = {str(u): {str(i): v for i, v in items.items()} for u, items in source_info.items()}
    ground_truth = val_ground_truth.set_index('customer_id')['purchased_articles'].to_dict()
    str_ground_truth = {str(k): set([str(x) for x in v]) for k, v in ground_truth.items()}

    samples = []
    for user_id in tqdm(val_customers['customer_id'], desc="构造样本"):
        if user_id not in recall_recs:
            continue

        safe_u = str(user_id)
        purchased_items = str_ground_truth.get(safe_u, set())

        for item_id in recall_recs[user_id]:
            safe_i = str(item_id)
            source = str_source_info.get(safe_u, {}).get(safe_i, {})
            samples.append({
                'customer_id': user_id,
                'article_id': item_id,
                'purchased': 1 if safe_i in purchased_items else 0,
                'is_from_repurchase': source.get('is_from_repurchase', 0),
                'is_from_itemcf': source.get('is_from_itemcf', 0),
                'is_from_popularity': source.get('is_from_popularity', 0),
                'repurchase_rank': source.get('repurchase_rank', 999),
                'itemcf_rank': source.get('itemcf_rank', 999)
            })

    train_df = pd.DataFrame(samples)
    train_df = train_df.sort_values('customer_id').reset_index(drop=True)
    return train_df


def batch_predict_and_save(
    ranker,
    feature_cols,
    all_customers_df,
    uint_to_hex_cust,
    all_recall_recs,
    all_source_info,
    full_transactions,
    raw_customers,
    raw_articles,
    output_path
):
    print("\n🎯 开始全量分批预测 (边推边落盘)...", flush=True)
    all_users = all_customers_df['customer_id'].unique()
    batch_size = 20000

    global_popular_str = [
        str(x).zfill(10)
        for x in list(full_transactions['article_id'].value_counts().index[:12])
    ]

    with open(output_path, 'w') as f:
        f.write("customer_id,prediction\n")
        for i in tqdm(range(0, len(all_users), batch_size), desc="分批推断"):
            batch_users = all_users[i:i + batch_size]

            samples = []
            for user_id in batch_users:
                recs = all_recall_recs.pop(user_id, [])
                if not recs:
                    continue

                u_source = all_source_info.pop(user_id, {})
                for item_id in recs:
                    source = u_source.get(item_id, {})
                    samples.append({
                        'customer_id': user_id,
                        'article_id': item_id,
                        'is_from_repurchase': source.get('is_from_repurchase', 0),
                        'is_from_itemcf': source.get('is_from_itemcf', 0),
                        'is_from_popularity': source.get('is_from_popularity', 0),
                        'repurchase_rank': source.get('repurchase_rank', 999),
                        'itemcf_rank': source.get('itemcf_rank', 999)
                    })

            if samples:
                batch_df = pd.DataFrame(samples)
                batch_df = extract_advanced_features_for_twostage(
                    batch_df,
                    full_transactions,
                    raw_customers,
                    raw_articles
                )

                batch_df['score'] = ranker.predict(batch_df[feature_cols])

                res = batch_df[['customer_id', 'article_id', 'score']].sort_values(
                    ['customer_id', 'score'],
                    ascending=[True, False]
                )
                top_items_df = res.groupby('customer_id').head(12)
                pred_dict = top_items_df.groupby('customer_id')['article_id'].apply(list).to_dict()

                for uid in batch_users:
                    u_hex = uint_to_hex_cust.get(uid, str(uid))
                    recs = pred_dict.get(uid, [])
                    recs_str = [str(x).zfill(10) for x in recs]
                    if len(recs_str) < 12:
                        recs_str.extend([x for x in global_popular_str if x not in recs_str])
                    f.write(f"{u_hex},{' '.join(recs_str[:12])}\n")

                del batch_df, samples, res, top_items_df, pred_dict
            else:
                for uid in batch_users:
                    u_hex = uint_to_hex_cust.get(uid, str(uid))
                    f.write(f"{u_hex},{' '.join(global_popular_str)}\n")

            f.flush()
            gc.collect()


def main():
    transactions, customers, uint_to_hex_cust, raw_customers, raw_articles = load_full_data()

    print("\n--- 阶段一：准备训练数据 ---")
    train_trans, val_trans, val_customers_df = split_data_by_time(transactions, customers, 1)
    val_truth = prepare_validation_data(val_trans)

    mgr = RecallManager(
        train_trans=train_trans,
        val_customers=val_customers_df,
        top_n=150,
        itemcf_top_k=150,
        max_items=5000,
        recall_cutoff=100
    )
    recs, source = mgr.multi_recall_with_source_info()

    train_df = create_ranking_samples_full(val_customers_df, val_truth, recs, source)
    train_df = extract_advanced_features_for_twostage(train_df, train_trans, raw_customers, raw_articles)

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

    print("🔥 训练 LightGBM...")
    params = {
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
        train_df,
        feature_cols,
        params,
        categorical_cols=cat_columns
    )

    del train_df, train_trans, val_trans, val_customers_df, val_truth, mgr, recs, source
    gc.collect()
    print_memory_usage("训练完成，内存已释放")

    print("\n--- 阶段二：全量推理 (预测未来一周) ---")
    infer_mgr = RecallManager(
        train_trans=transactions,
        val_customers=customers,
        top_n=150,
        itemcf_top_k=150,
        max_items=5000,
        recall_cutoff=100
    )
    all_recs, all_source = infer_mgr.multi_recall_with_source_info()

    batch_predict_and_save(
        ranker,
        feature_cols,
        customers,
        uint_to_hex_cust,
        all_recs,
        all_source,
        transactions,
        raw_customers,
        raw_articles,
        'submission.csv'
    )
    print("🎉 全量跑批完成，结果已保存至 submission.csv！")


if __name__ == "__main__":
    main()