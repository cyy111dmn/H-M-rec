#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def create_radek_dataset(transactions, customers, articles):
    """Radek 官方单阶段召回+排序数据集构造逻辑"""
    print("计算时间窗口 (Weeks)...", flush=True)
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    max_date = transactions['t_dat'].max()
    transactions['week'] = (max_date - transactions['t_dat']).dt.days // 7
    transactions = transactions[transactions['week'] < 10].copy()

    print("提取每周爆款并计算特征...", flush=True)
    weekly_sales = transactions.groupby(['week', 'article_id']).size().reset_index(name='sales_count')
    bestsellers_prev_week = weekly_sales.sort_values(['week', 'sales_count'], ascending=[True, False])
    bestsellers_prev_week = bestsellers_prev_week.groupby('week').head(12).reset_index(drop=True)
    bestsellers_prev_week['bestseller_rank'] = bestsellers_prev_week.groupby('week').cumcount() + 1
    bestsellers_prev_week['week'] -= 1

    transactions['purchased'] = 1
    unique_users_per_week = transactions.groupby(['week', 'customer_id']).head(1)[['week', 'customer_id']]
    negative_candidates = pd.merge(
        unique_users_per_week,
        bestsellers_prev_week[['week', 'article_id']],
        on='week'
    )
    negative_candidates['purchased'] = 0

    print("拼接正负样本并去重...", flush=True)
    train_df = pd.concat([
        transactions[['customer_id', 'article_id', 'week', 'purchased']],
        negative_candidates
    ])
    train_df.drop_duplicates(subset=['customer_id', 'article_id', 'week'], keep='first', inplace=True)

    train_df = pd.merge(
        train_df,
        bestsellers_prev_week[['week', 'article_id', 'bestseller_rank', 'sales_count']],
        on=['week', 'article_id'],
        how='left'
    )
    train_df['bestseller_rank'] = train_df['bestseller_rank'].fillna(999).astype(np.int16)
    train_df['sales_count'] = train_df['sales_count'].fillna(0).astype(np.float32)

    user_features = ['customer_id', 'age', 'club_member_status', 'fashion_news_frequency']
    train_df = pd.merge(train_df, customers[user_features], on='customer_id', how='left')

    item_features = ['article_id', 'product_group_name', 'index_group_name', 'colour_group_name', 'graphical_appearance_name']
    train_df = pd.merge(train_df, articles[item_features], on='article_id', how='left')

    first_purchase = transactions.groupby(['customer_id', 'article_id'])['week'].max().reset_index(name='first_buy_week')
    train_df = pd.merge(train_df, first_purchase, on=['customer_id', 'article_id'], how='left')
    train_df['is_repurchase'] = (train_df['first_buy_week'] > train_df['week']).astype(np.int8)
    train_df.drop(columns=['first_buy_week'], inplace=True)

    cat_columns = [
        'club_member_status', 'fashion_news_frequency',
        'product_group_name', 'index_group_name',
        'colour_group_name', 'graphical_appearance_name'
    ]
    for col in cat_columns:
        train_df[col] = train_df[col].astype('category')

    test_bestsellers = bestsellers_prev_week[bestsellers_prev_week['week'] == -1].copy()
    return train_df, transactions, test_bestsellers


def _merge_customer_item_profiles(train_df, customers, articles):
    """拼接用户和商品画像（不依赖 train_trans，可安全分批调用）。"""
    customers = customers.copy()
    if 'club_member_status' in customers.columns:
        customers['club_member_status'] = customers['club_member_status'].fillna('NONE')
    if 'fashion_news_frequency' in customers.columns:
        customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna('NONE')

    user_cols = ['customer_id', 'age', 'club_member_status', 'fashion_news_frequency']
    user_cols = [c for c in user_cols if c in customers.columns]
    train_df = pd.merge(train_df, customers[user_cols], on='customer_id', how='left')

    item_cols = ['article_id', 'product_group_name', 'index_group_name', 'colour_group_name', 'graphical_appearance_name']
    item_cols = [c for c in item_cols if c in articles.columns]
    train_df = pd.merge(train_df, articles[item_cols], on='article_id', how='left')
    return train_df


def _precompute_and_apply_features(train_df, train_trans, customers, articles):
    """一次性预计算 + 应用特征（兼容旧式单次调用）。"""
    lookups = precompute_feature_lookups(train_trans, articles)
    train_df = _merge_customer_item_profiles(train_df, customers, articles)
    train_df = apply_feature_lookups(train_df, lookups, articles)
    return train_df


def precompute_feature_lookups(train_trans, articles):
    """预计算所有基于 train_trans 的查找表，避免分批预测时重复计算。"""
    print("📦 预计算特征查找表...", flush=True)
    lookups = {}

    lookups['item_sales'] = train_trans['article_id'].value_counts().to_dict()
    lookups['user_activity'] = train_trans['customer_id'].value_counts().to_dict()
    lookups['historical_pairs'] = set(zip(train_trans['customer_id'], train_trans['article_id']))

    # RFM: Recency
    train_trans_ = train_trans.copy()
    train_trans_['t_dat'] = pd.to_datetime(train_trans_['t_dat'])
    max_date = train_trans_['t_dat'].max()
    user_last_purchase = train_trans_.groupby('customer_id')['t_dat'].max()
    lookups['user_recency_days'] = (max_date - user_last_purchase).dt.days  # 越小越活跃

    # RFM: Monetary (total spend)
    if 'price' in train_trans_.columns:
        user_monetary = train_trans_.groupby('customer_id')['price'].sum()
        lookups['user_monetary'] = user_monetary
        user_avg_basket = train_trans_.groupby('customer_id')['price'].mean()
        lookups['user_avg_basket'] = user_avg_basket

    del train_trans_

    if 'price' in train_trans.columns:
        lookups['item_avg_price'] = train_trans.groupby('article_id')['price'].mean().to_dict()
        lookups['user_price_mean'] = train_trans.groupby('customer_id')['price'].mean()
        lookups['user_price_p50'] = train_trans.groupby('customer_id')['price'].median()
        lookups['user_price_p10'] = train_trans.groupby('customer_id')['price'].quantile(0.1)
        lookups['user_price_p90'] = train_trans.groupby('customer_id')['price'].quantile(0.9)

    train_trans_week = train_trans.copy()
    train_trans_week['t_dat'] = pd.to_datetime(train_trans_week['t_dat'])
    max_date = train_trans_week['t_dat'].max()
    train_trans_week['week'] = (max_date - train_trans_week['t_dat']).dt.days // 7
    lookups['max_week'] = train_trans_week['week'].max()
    lookups['item_first_week'] = train_trans_week.groupby('article_id')['week'].max().to_dict()
    lookups['current_week'] = 0

    hist_df = train_trans[['customer_id', 'article_id']].merge(
        articles[['article_id', 'index_group_name', 'colour_group_name']],
        on='article_id', how='inner'
    )
    top_cat = hist_df.groupby(['customer_id', 'index_group_name']).size().reset_index(name='count')
    top_cat = top_cat.sort_values(['customer_id', 'count'], ascending=[True, False]).drop_duplicates('customer_id')
    lookups['top_cat_dict'] = dict(zip(top_cat['customer_id'], top_cat['index_group_name']))

    top_col = hist_df.groupby(['customer_id', 'colour_group_name']).size().reset_index(name='count')
    top_col = top_col.sort_values(['customer_id', 'count'], ascending=[True, False]).drop_duplicates('customer_id')
    lookups['top_col_dict'] = dict(zip(top_col['customer_id'], top_col['colour_group_name']))

    del train_trans_week, hist_df, top_cat, top_col
    return lookups


def apply_feature_lookups(train_df, lookups, articles):
    """使用预计算的查找表快速提取特征（不重复计算 train_trans 统计量）。"""
    train_df['item_popularity'] = train_df['article_id'].map(lookups['item_sales']).fillna(0).astype(np.float32)
    train_df['user_activity'] = train_df['customer_id'].map(lookups['user_activity']).fillna(0).astype(np.float32)
    train_df['user_activity_log'] = np.log1p(train_df['user_activity']).astype(np.float32)
    bucket = pd.qcut(train_df['user_activity'].rank(method='first'), q=5, labels=False, duplicates='drop')
    train_df['user_activity_bucket'] = bucket.fillna(0).astype(np.int8)

    # RFM 特征
    train_df['user_recency_days'] = train_df['customer_id'].map(lookups['user_recency_days']).fillna(999).astype(np.int16)
    if 'user_monetary' in lookups:
        train_df['user_monetary'] = train_df['customer_id'].map(lookups['user_monetary']).fillna(0).astype(np.float32)
        train_df['user_avg_basket'] = train_df['customer_id'].map(lookups['user_avg_basket']).fillna(0).astype(np.float32)

    hp = lookups['historical_pairs']
    train_df['is_repurchase'] = [1 if (u, i) in hp else 0 for u, i in zip(train_df['customer_id'], train_df['article_id'])]
    train_df['is_repurchase'] = train_df['is_repurchase'].astype(np.int8)

    if 'item_avg_price' in lookups:
        train_df['item_price'] = train_df['article_id'].map(lookups['item_avg_price']).fillna(0).astype(np.float32)
        train_df['user_avg_price'] = train_df['customer_id'].map(lookups['user_price_mean']).fillna(0).astype(np.float32)
        train_df['user_price_p50'] = train_df['customer_id'].map(lookups['user_price_p50']).fillna(0).astype(np.float32)
        train_df['user_price_p10'] = train_df['customer_id'].map(lookups['user_price_p10']).fillna(0).astype(np.float32)
        train_df['user_price_p90'] = train_df['customer_id'].map(lookups['user_price_p90']).fillna(0).astype(np.float32)
        train_df['price_diff'] = np.abs(train_df['user_avg_price'] - train_df['item_price']).astype(np.float32)
    else:
        for c in ['item_price', 'user_avg_price', 'user_price_p50', 'user_price_p10', 'user_price_p90', 'price_diff']:
            train_df[c] = 0.0

    item_first_week = lookups['item_first_week']
    train_df['item_first_week'] = train_df['article_id'].map(item_first_week).fillna(0)
    train_df['item_age_weeks'] = (train_df['item_first_week'] - lookups['current_week']).astype(np.int16)
    train_df.drop(columns=['item_first_week'], inplace=True)

    NEW_ITEM_THRESHOLD = 4
    train_df['is_new_repurchase'] = ((train_df['item_age_weeks'] < NEW_ITEM_THRESHOLD) & (train_df['is_repurchase'] == 1)).astype(np.int8)

    top_cat_dict, top_col_dict = lookups['top_cat_dict'], lookups['top_col_dict']
    train_df['is_category_matched'] = (train_df['index_group_name'] == train_df['customer_id'].map(top_cat_dict)).astype(np.int8)
    train_df['is_color_matched'] = (train_df['colour_group_name'] == train_df['customer_id'].map(top_col_dict)).astype(np.int8)

    cat_columns = ['club_member_status', 'fashion_news_frequency', 'product_group_name',
                   'index_group_name', 'colour_group_name', 'graphical_appearance_name']
    for col in cat_columns:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype('category')

    return train_df


def extract_advanced_features_for_twostage(train_df, train_trans, customers, articles):
    """
    专门为【多路召回->精排】双阶段架构打造的特征提取函数
    融合了：基础统计、偏好命中、价格匹配、商品生命周期
    """
    return _precompute_and_apply_features(train_df, train_trans, customers, articles)