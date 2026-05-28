#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import os
import time

import numpy as np
import pandas as pd


def main():
    np.random.seed(42)

    data_dir = '/root/autodl-tmp/hm_recommender/hm_recommand/data'
    output_dir = './offline_data'
    os.makedirs(output_dir, exist_ok=True)

    print("📂 1. 加载全量交易数据...")
    start_time = time.time()
    trans = pd.read_csv(os.path.join(data_dir, 'transactions_train.csv'))
    trans['t_dat'] = pd.to_datetime(trans['t_dat'])
    print(f"   ✅ 数据加载完毕！耗时: {time.time() - start_time:.1f} 秒")

    # 严格按时间切分：最后 7 天作为验证期
    max_date = trans['t_dat'].max()
    val_start_date = max_date - pd.Timedelta(days=6)

    print(f"\n🔪 2. 按时间切分数据 (验证期起始点: {val_start_date.date()})")
    start_time = time.time()

    val_trans = trans[trans['t_dat'] >= val_start_date].copy()
    train_trans = trans[trans['t_dat'] < val_start_date].copy()

    del trans
    gc.collect()
    print(f"   ✅ 切分完毕！耗时: {time.time() - start_time:.1f} 秒")

    print("\n🎯 3. 构造验证集目标用户...")
    start_time = time.time()

    active_users = val_trans['customer_id'].unique()
    sampled_active = np.random.choice(
        active_users,
        size=min(40000, len(active_users)),
        replace=False
    )

    history_users = train_trans['customer_id'].unique()
    cold_users = list(set(history_users) - set(active_users))
    sampled_cold = np.random.choice(
        cold_users,
        size=min(10000, len(cold_users)),
        replace=False
    ) if len(cold_users) > 0 else np.array([])

    val_users = np.concatenate([sampled_active, sampled_cold])
    val_users = pd.Series(val_users).drop_duplicates().values

    print(f"   - 抽样评估用户总计: {len(val_users)} 人")
    print(f"   ✅ 抽样完毕！耗时: {time.time() - start_time:.1f} 秒")

    print("\n💾 4. 提取数据并落盘...")
    start_time = time.time()

    offline_train_trans = train_trans.copy()

    print("   ⚙️ 正在将真实购买流水聚合成用户列表格式...")
    offline_val_truth = val_trans[val_trans['customer_id'].isin(val_users)].copy()
    offline_val_truth = (
        offline_val_truth.groupby('customer_id')['article_id']
        .apply(list)
        .reset_index()
        .rename(columns={'article_id': 'purchased_articles'})
    )

    # 关键修复：把所有验证用户都补齐，没买过的用户标签设为空列表
    offline_users_df = pd.DataFrame({'customer_id': val_users})
    offline_val_truth = offline_users_df.merge(
        offline_val_truth,
        on='customer_id',
        how='left'
    )
    offline_val_truth['purchased_articles'] = offline_val_truth['purchased_articles'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    offline_train_trans.to_parquet(
        os.path.join(output_dir, 'offline_train_trans.parquet'),
        index=False
    )
    offline_val_truth.to_parquet(
        os.path.join(output_dir, 'offline_val_truth.parquet'),
        index=False
    )
    offline_users_df.to_parquet(
        os.path.join(output_dir, 'offline_users.parquet'),
        index=False
    )

    print(f"   ✅ 提取并落盘完毕！耗时: {time.time() - start_time:.1f} 秒")
    print(f"\n🎉 大功告成！全流程结束，请前往 {output_dir}/ 查收文件。")


if __name__ == "__main__":
    main()