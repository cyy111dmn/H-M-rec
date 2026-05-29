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


def extract_advanced_features_for_twostage(train_df, train_trans, customers, articles):
    """
    专门为【多路召回->精排】双阶段架构打造的特征提取函数
    融合了：基础统计、偏好命中、价格匹配、商品生命周期
    """
    print("📊 [特征工程] 提取基础统计特征...", flush=True)

    item_sales = train_trans['article_id'].value_counts().to_dict()
    train_df['item_popularity'] = train_df['article_id'].map(item_sales).fillna(0).astype(np.float32)

    user_activity = train_trans['customer_id'].value_counts().to_dict()
    train_df['user_activity'] = train_df['customer_id'].map(user_activity).fillna(0).astype(np.float32)

    train_df['user_activity_log'] = np.log1p(train_df['user_activity']).astype(np.float32)
    train_df['user_activity_bucket'] = pd.qcut(
        train_df['user_activity'].rank(method='first'),
        q=5, labels=False, duplicates='drop'
    ).astype(np.int8)

    print("🔄 [特征工程] 计算历史复购特征 (极速省内存版)...", flush=True)
    historical_pairs = set(zip(train_trans['customer_id'], train_trans['article_id']))
    train_df['is_repurchase'] = [
        1 if (u, i) in historical_pairs else 0
        for u, i in zip(train_df['customer_id'], train_df['article_id'])
    ]
    train_df['is_repurchase'] = train_df['is_repurchase'].astype(np.int8)

    print("💰 [特征工程] 计算价格敏感度匹配（含分位数）...", flush=True)
    if 'price' in train_trans.columns:
        item_avg_price = train_trans.groupby('article_id')['price'].mean().to_dict()
        train_df['item_price'] = train_df['article_id'].map(item_avg_price).fillna(0).astype(np.float32)

        user_price_stats = train_trans.groupby('customer_id')['price'].agg(['mean', 'median', lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)]).to_dict('index')
        train_df['user_avg_price'] = train_df['customer_id'].map(lambda u: user_price_stats.get(u, {}).get('mean', 0)).astype(np.float32)
        train_df['user_price_p50'] = train_df['customer_id'].map(lambda u: user_price_stats.get(u, {}).get('median', 0)).astype(np.float32)
        train_df['user_price_p10'] = train_df['customer_id'].map(lambda u: user_price_stats.get(u, {}).get('<lambda_0>', 0)).astype(np.float32)
        train_df['user_price_p90'] = train_df['customer_id'].map(lambda u: user_price_stats.get(u, {}).get('<lambda_1>', 0)).astype(np.float32)

        train_df['price_diff'] = np.abs(train_df['user_avg_price'] - train_df['item_price']).astype(np.float32)
    else:
        train_df['item_price'] = 0.0
        train_df['user_avg_price'] = 0.0
        train_df['user_price_p50'] = 0.0
        train_df['user_price_p10'] = 0.0
        train_df['user_price_p90'] = 0.0
        train_df['price_diff'] = 0.0

    print("⏳ [特征工程] 计算商品生命周期 (Item Age)...", flush=True)
    if 'week' not in train_trans.columns and 't_dat' in train_trans.columns:
        train_trans = train_trans.copy()
        train_trans['t_dat'] = pd.to_datetime(train_trans['t_dat'])
        max_date = train_trans['t_dat'].max()
        train_trans['week'] = (max_date - train_trans['t_dat']).dt.days // 7

    if 'week' in train_trans.columns:
        item_first_week = train_trans.groupby('article_id')['week'].max().to_dict()
        train_df['item_first_week'] = train_df['article_id'].map(item_first_week).fillna(0)
        current_week = train_df['week'] if 'week' in train_df.columns else 0
        train_df['item_age_weeks'] = (train_df['item_first_week'] - current_week).astype(np.int16)
        train_df.drop(columns=['item_first_week'], inplace=True)
    else:
        train_df['item_age_weeks'] = 0

    print("🆕 [特征工程] 计算新品+复购交叉特征...", flush=True)
    NEW_ITEM_THRESHOLD = 4  # 最近 4 周内上架算"新品"
    train_df['is_new_repurchase'] = (
        (train_df['item_age_weeks'] < NEW_ITEM_THRESHOLD) & (train_df['is_repurchase'] == 1)
    ).astype(np.int8)

    print("👤 [特征工程] 拼接画像与偏好交叉特征...", flush=True)
    hist_df = train_trans[['customer_id', 'article_id']].merge(
        articles[['article_id', 'index_group_name', 'colour_group_name']],
        on='article_id',
        how='inner'
    )

    top_cat = hist_df.groupby(['customer_id', 'index_group_name']).size().reset_index(name='count')
    top_cat = top_cat.sort_values(['customer_id', 'count'], ascending=[True, False]).drop_duplicates('customer_id')
    top_cat_dict = dict(zip(top_cat['customer_id'], top_cat['index_group_name']))

    top_col = hist_df.groupby(['customer_id', 'colour_group_name']).size().reset_index(name='count')
    top_col = top_col.sort_values(['customer_id', 'count'], ascending=[True, False]).drop_duplicates('customer_id')
    top_col_dict = dict(zip(top_col['customer_id'], top_col['colour_group_name']))

    del hist_df, top_cat, top_col

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

    train_df['is_category_matched'] = (
        train_df['index_group_name'] == train_df['customer_id'].map(top_cat_dict)
    ).astype(np.int8)

    train_df['is_color_matched'] = (
        train_df['colour_group_name'] == train_df['customer_id'].map(top_col_dict)
    ).astype(np.int8)

    print("🔄 [特征工程] 转换类别特征 (Category)...", flush=True)
    cat_columns = [
        'club_member_status', 'fashion_news_frequency',
        'product_group_name', 'index_group_name',
        'colour_group_name', 'graphical_appearance_name'
    ]
    for col in cat_columns:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype('category')

    return train_df