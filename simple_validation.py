#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础召回策略验证脚本 (进阶版)
对比：热门Baseline vs 多路召回（热门+复购） vs LGBM精排(引入画像特征)
"""

import pandas as pd
import numpy as np
import os
import sys
import psutil
import gc
import time

# 添加 src 目录到路径
sys.path.append('src')

from src.metrics import calculate_map_at_k
from src.features import extract_advanced_features_for_twostage

def load_real_data():
    """加载交易数据以及用户和商品画像"""
    print("📂 加载基础数据与画像表...", flush=True)
    
    # ⚠️ 确保这些路径和你的 AutoDL 实际路径一致！
    trans_path = '/root/autodl-tmp/data/interim/val_1000_users.csv'
    raw_dir = '/root/autodl-tmp/hm_recommender/hm_recommand/data'
    
    try:
        transactions = pd.read_csv(trans_path)
        # 读取完整的 user 和 item 信息
        customers = pd.read_csv(f'{raw_dir}/customers.csv')
        articles = pd.read_csv(f'{raw_dir}/articles.csv')
        
        # 为了极速验证，过滤一下customers，只保留这1000人的画像，省内存
        valid_users = transactions['customer_id'].unique()
        customers = customers[customers['customer_id'].isin(valid_users)].copy()
        
        print(f"✅ 成功加载: {len(customers)} 用户画像, {len(articles)} 商品画像, {len(transactions)} 交易记录")
        return transactions, customers, articles
    except FileNotFoundError as e:
        print(f"❌ 数据文件不存在: {e}")
        return None, None, None

def create_validation_data(transactions, validation_ratio=0.2):
    print("🔪 创建验证数据...", flush=True)
    validation_data, training_data = [], []
    for user_id, user_trans in transactions.groupby('customer_id'):
        user_trans_sorted = user_trans.sort_values('t_dat', ascending=False)
        split_point = int(len(user_trans_sorted) * validation_ratio)
        validation_data.append(user_trans_sorted.head(split_point))
        training_data.append(user_trans_sorted.tail(len(user_trans_sorted) - split_point))
        
    train_trans = pd.concat(training_data, ignore_index=True)
    val_trans = pd.concat(validation_data, ignore_index=True)
    
    val_customers = pd.DataFrame({'customer_id': val_trans['customer_id'].unique()})
    val_purchases = val_trans.groupby('customer_id')['article_id'].apply(list).reset_index()
    val_purchases.columns = ['customer_id', 'purchased_articles']
    
    val_customers = val_customers.merge(val_purchases, on='customer_id', how='left')
    val_ground_truth = val_customers[['customer_id', 'purchased_articles']]
    
    return train_trans, val_customers, val_ground_truth

def run_popularity_baseline(train_trans, val_customers, val_ground_truth):
    print("\n" + "="*60 + "\n🔥 热门Baseline实验\n" + "="*60)
    start_time = time.time()
    
    train_trans['t_dat'] = pd.to_datetime(train_trans['t_dat'])
    max_date = train_trans['t_dat'].max()
    train_trans['week'] = (max_date - train_trans['t_dat']).dt.days // 7
    
    last_week_data = train_trans[train_trans['week'] == train_trans['week'].min()]
    popular_items = last_week_data.groupby('article_id').size().sort_values(ascending=False).head(50).index.tolist()
    
    predicted_dict = {user_id: popular_items[:12] for user_id in val_customers['customer_id']}
    map_score = calculate_map_at_k(val_ground_truth.set_index('customer_id')['purchased_articles'].to_dict(), predicted_dict, k=12)
    
    print(f"🎯 热门Baseline MAP@12: {map_score:.6f}")
    return {'map_at_k': map_score}

def run_multi_recall_with_source(train_trans, val_customers, val_ground_truth):
    print("\n" + "="*60 + "\n🎯 多路召回实验（带来源信息）\n" + "="*60)
    from src.recall_merged import RecallManager
    
    recall_manager = RecallManager(train_trans=train_trans, val_customers=val_customers, top_n=50, itemcf_top_k=50, max_items=5000, recall_cutoff=50)
    final_recs, source_info = recall_manager.multi_recall_with_source_info()
    
    map_score = calculate_map_at_k(val_ground_truth.set_index('customer_id')['purchased_articles'].to_dict(), final_recs, k=12)
    print(f"🎯 多路召回 MAP@12: {map_score:.6f}")
    return {'map_at_k': map_score}, final_recs, source_info

def create_ranking_samples(val_customers, val_ground_truth, recall_recs, source_info=None):
    print("📋 构造精排样本表...", flush=True)
    str_source_info = {}
    if source_info:
        for u, items_dict in source_info.items():
            str_source_info[str(u)] = {str(i): v for i, v in items_dict.items()}
            
    ground_truth = val_ground_truth.set_index('customer_id')['purchased_articles'].to_dict()
    str_ground_truth = {str(k): set([str(x) for x in v]) for k, v in ground_truth.items()}
    
    samples = []
    for user_id in val_customers['customer_id']:
        if user_id not in recall_recs: continue
        safe_u = str(user_id)
        purchased_items = str_ground_truth.get(safe_u, set())
        
        for item_id in recall_recs[user_id]:
            safe_i = str(item_id)
            sample = {'customer_id': user_id, 'article_id': item_id, 'purchased': 1 if safe_i in purchased_items else 0}
            
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
                sample.update({'is_from_repurchase': 0, 'is_from_itemcf': 0, 'is_from_popularity': 0, 'repurchase_rank': 999, 'itemcf_rank': 999})
            samples.append(sample)
            
    train_df = pd.DataFrame(samples)
    print(f"✅ 样本表构造完成: {len(train_df)} 行 | 正样本: {train_df['purchased'].sum()}")
    return train_df

def run_lgbm_ranking_experiment(train_trans, val_customers, val_ground_truth, recall_recs, source_info, customers_raw, articles_raw):
    print("\n" + "="*60 + "\n🎯 LGBM 精排实验 (含高级画像特征)\n" + "="*60)
    from src.ranker import train_lgbm_ranker
    
    # 1. 构造正负样本
    train_df = create_ranking_samples(val_customers, val_ground_truth, recall_recs, source_info)
    
    # 2. 调用新写好的高级特征提取库
    train_df = extract_advanced_features_for_twostage(train_df, train_trans, customers_raw, articles_raw)
    
    # 3. 扩充特征列表，把所有画像特征加进去！
    feature_cols = [
        'item_popularity', 'user_activity', 'is_repurchase', 
        'is_from_repurchase', 'is_from_itemcf', 'is_from_popularity', 
        'repurchase_rank', 'itemcf_rank',
        'age', 'club_member_status', 'fashion_news_frequency', # 用户特征
        'product_group_name', 'index_group_name', 'colour_group_name', 'graphical_appearance_name' # 商品特征
    ]
    
    print("🚀 训练 LGBM 模型...", flush=True)
    lgbm_params = {'objective': 'lambdarank', 'metric': 'map', 'learning_rate': 0.1, 'num_leaves': 31, 'verbose': -1, 'random_state': 42}
    ranker = train_lgbm_ranker(train_df, feature_cols, lgbm_params)
    
    print("🎯 进行预测打分...")
    # 向量化批量打分
    train_df['score'] = ranker.predict(train_df[feature_cols])
    
    # 获取Top12
    sorted_df = train_df[['customer_id', 'article_id', 'score']].sort_values(['customer_id', 'score'], ascending=[True, False])
    top_items_df = sorted_df.groupby('customer_id').head(12)
    ranked_recs = top_items_df.groupby('customer_id')['article_id'].apply(list).to_dict()
    
    for user_id in val_customers['customer_id']:
        if user_id not in ranked_recs: ranked_recs[user_id] = []
            
    map_score = calculate_map_at_k(val_ground_truth.set_index('customer_id')['purchased_articles'].to_dict(), ranked_recs, k=12)
    print(f"🎯 LGBM 精排 MAP@12: {map_score:.6f}")
    return {'map_at_k': map_score}

def main():
    print("🚀 开始验证召回 -> 排序 双阶段架构")
    # 这次返回了三个表
    transactions, customers, articles = load_real_data()
    if transactions is None: return
    
    train_trans, val_customers, val_ground_truth = create_validation_data(transactions)
    baseline_results = run_popularity_baseline(train_trans, val_customers, val_ground_truth)
    
    multi_recall_results, multi_recs, source_info = run_multi_recall_with_source(train_trans, val_customers, val_ground_truth)
    
    # 将真实的 customer 和 article 数据传入精排函数
    lgbm_results = run_lgbm_ranking_experiment(train_trans, val_customers, val_ground_truth, multi_recs, source_info, customers, articles)
    
    print("\n" + "="*60 + "\n🏆 最终战报 (纯召回 vs 加入多维特征精排)\n" + "="*60)
    print(f"多路召回 (MAP@12):  {multi_recall_results['map_at_k']:.6f}")
    print(f"LGBM精排 (MAP@12):  {lgbm_results['map_at_k']:.6f}")
    improvement = (lgbm_results['map_at_k'] - multi_recall_results['map_at_k']) / multi_recall_results['map_at_k'] * 100
    print(f"📈 精排层相对召回层提升: {improvement:+.2f}%")

if __name__ == "__main__":
    main()