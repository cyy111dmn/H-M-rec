#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量跑批引擎 — 多路召回 → 生成 Kaggle 提交文件

策略：纯召回（不用 Ranker），因为当前 Ranker 训练数据太少导致线上分数下降。
      等 Ranker 训练数据量和特征质量足够后，再切回双阶段。
"""

import gc
import os
import sys
import time
import argparse

import pandas as pd
import psutil

sys.path.append('src')

from src.data_loader import load_id_mapping
from src.recall_merged import RecallManager


def print_memory_usage(step):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"[内存监控] {step}: {mem:.2f} MB", flush=True)


def load_full_data():
    """加载全量 CSV 数据 + ID 映射"""
    print("📂 加载全量数据...", flush=True)
    data_dir = './data'

    # 全量数据只有 CSV 格式（3.3GB），parquet 是采样数据
    transactions = pd.read_csv(f'{data_dir}/transactions_train.csv')
    customers = pd.read_csv(f'{data_dir}/customers.csv')[['customer_id']]

    uint_to_hex_cust = load_id_mapping(data_dir)

    print(f"📊 {len(customers):,} 用户, {len(transactions):,} 交易记录")
    print_memory_usage("数据加载完成")
    return transactions, customers, uint_to_hex_cust


def generate_submission(recs, all_customers, uint_to_hex_cust, output_dir='submissions', exp_name='recall'):
    """将召回结果转为 Kaggle 提交文件"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'submission_{exp_name}.csv')

    print(f"\n📝 生成提交文件: {output_path}", flush=True)
    all_users = all_customers['customer_id'].unique()
    start = time.time()

    # 全局热门 top-12 用于兜底
    global_popular = list(recs[list(recs.keys())[0]][:12]) if recs else []

    with open(output_path, 'w') as f:
        f.write("customer_id,prediction\n")
        for uid in all_users:
            u_hex = uint_to_hex_cust.get(uid, str(uid))
            items = recs.get(uid, [])[:12]
            if len(items) < 12:
                seen = set(items)
                items.extend(x for x in global_popular if x not in seen)
            line = f"{u_hex},{' '.join(str(x).zfill(10) for x in items[:12])}\n"
            f.write(line)

    elapsed = time.time() - start
    print(f"✅ 提交文件已保存: {output_path}")
    print(f"⏱️  耗时: {elapsed:.1f} 秒")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='recall',
                        help='Experiment name, used in output filename')
    args = parser.parse_args()
    exp_name = args.exp_name

    print("=" * 60)
    print(f"🚀 全量跑批 — 多路召回 ({exp_name})")
    print("=" * 60)

    # 1. 加载数据
    transactions, customers, uint_to_hex_cust = load_full_data()
    print_memory_usage("数据加载完成")

    # 2. 多路召回
    print("\n--- 执行多路召回 ---")
    mgr = RecallManager(
        train_trans=transactions,
        val_customers=customers,
        top_n=150,
        itemcf_top_k=150,
        max_items=5000,
        recall_cutoff=100,
    )
    final_recs = mgr.multi_recall()
    print_memory_usage("召回完成")

    del mgr
    gc.collect()

    # 3. 生成提交文件
    generate_submission(final_recs, customers, uint_to_hex_cust, exp_name=exp_name)

    # 4. 统计
    total_items = sum(len(v) for v in final_recs.values())
    total_users = len(final_recs)
    print(f"\n📊 统计: {total_users:,} 用户, {total_items:,} 推荐商品")
    print(f"🎉 完成！将 submissions/submission_{exp_name}.csv 提交到 Kaggle。")
