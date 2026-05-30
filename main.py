#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量跑批引擎 — 多路召回 → 可选精排 → 生成 Kaggle 提交文件
"""

import gc
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

sys.path.append('src')

from src.data_loader import load_id_mapping
from src.recall_merged import RecallManager
from src.features import extract_advanced_features_for_twostage, precompute_feature_lookups, apply_feature_lookups, _merge_customer_item_profiles
from src.utils import print_memory_usage


def load_full_data(with_profiles=False):
    print("📂 加载全量数据...", flush=True)
    data_dir = './data'
    transactions = pd.read_csv(f'{data_dir}/transactions_train.csv')
    uint_to_hex_cust = load_id_mapping(data_dir)
    if with_profiles:
        customers = pd.read_csv(f'{data_dir}/customers.csv')
        articles = pd.read_csv(f'{data_dir}/articles.csv')
    else:
        customers = pd.read_csv(f'{data_dir}/customers.csv')[['customer_id']]
        articles = None
    print(f"📊 {len(customers):,} 用户, {len(transactions):,} 交易记录")
    print_memory_usage("数据加载完成")
    return transactions, customers, articles, uint_to_hex_cust


def generate_submission(recs, all_customers, uint_to_hex_cust, output_dir='submissions', exp_name='recall'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'submission_{exp_name}.csv')
    print(f"\n📝 生成提交文件: {output_path}", flush=True)
    all_users = all_customers['customer_id'].unique()
    start = time.time()
    global_popular = list(recs[list(recs.keys())[0]][:12]) if recs else []
    with open(output_path, 'w') as f:
        f.write("customer_id,prediction\n")
        for uid in all_users:
            u_hex = uint_to_hex_cust.get(uid, str(uid))
            items = recs.get(uid, [])[:12]
            if len(items) < 12:
                seen = set(items)
                items.extend(x for x in global_popular if x not in seen)
            f.write(f"{u_hex},{' '.join(str(x).zfill(10) for x in items[:12])}\n")
    elapsed = time.time() - start
    print(f"✅ 提交文件已保存: {output_path}")
    print(f"⏱️  耗时: {elapsed:.1f} 秒")
    return output_path


def _build_source_for(recs, source_info, user_ids):
    str_src = {}
    for uid in user_ids:
        su = str(uid)
        src = source_info.get(su, source_info.get(uid, {}))
        if src:
            str_src[su] = {str(i): v for i, v in src.items()}
    return str_src


def _build_training_samples(recs, source_info, label_set, user_list):
    """构建训练样本，label_set 包含 (user_id, item_id) 为正样本。"""
    print("📋 构建训练样本...", flush=True)
    str_src = _build_source_for(recs, source_info, user_list)
    samples = []
    for uid in user_list:
        su = str(uid)
        if su not in recs:
            continue
        usrc = str_src.get(su, {})
        for item_id in recs[su]:
            si = str(item_id)
            src = usrc.get(si, {})
            samples.append({
                "customer_id": uid, "article_id": item_id,
                "purchased": 1 if (su, si) in label_set else 0,
                "is_from_repurchase": src.get("is_from_repurchase", 0),
                "is_from_itemcf": src.get("is_from_itemcf", 0),
                "is_from_usercf": src.get("is_from_usercf", 0),
                "is_from_w2v": src.get("is_from_w2v", 0),
                "is_from_popularity": src.get("is_from_popularity", 0),
                "repurchase_rank": src.get("repurchase_rank", 999),
                "itemcf_rank": src.get("itemcf_rank", 999),
                "usercf_rank": src.get("usercf_rank", 999),
                "w2v_rank": src.get("w2v_rank", 999),
            })
    return pd.DataFrame(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='recall')
    parser.add_argument('--use_ranker', action='store_true', help='启用 LGBM 精排')
    parser.add_argument('--use_deepfm', action='store_true', help='启用 DeepFM 精排')
    args = parser.parse_args()
    exp_name = args.exp_name

    if args.use_ranker:
        mode = "召回+LGBM精排"
    elif args.use_deepfm:
        mode = "召回+DeepFM精排"
    else:
        mode = "纯召回"

    print("=" * 60)
    print(f"🚀 全量跑批 — {mode} ({exp_name})")
    print("=" * 60)

    use_profiles = args.use_ranker or args.use_deepfm
    transactions, customers, articles, uint_to_hex_cust = load_full_data(with_profiles=use_profiles)
    print_memory_usage("数据加载完成")

    # 多路召回
    print("\n--- 执行多路召回 ---")
    mgr = RecallManager(
        train_trans=transactions,
        val_customers=customers,
        top_n=150, itemcf_top_k=150, usercf_top_k=20, w2v_top_k=20,
        max_items=5000, recall_cutoff=100,
    )
    final_recs, source_info = mgr.multi_recall_with_source_info()
    print_memory_usage("召回完成")

    # 保存 mgr.train_trans（已过滤到最近 27 周）用于训练标签
    train_trans = mgr.train_trans.copy()
    del mgr
    gc.collect()

    if args.use_ranker:
        # ===== LGBM 精排 =====
        # 标签：与召回同期 27 周内用户是否购买过该商品
        label_set = set(zip(
            train_trans['customer_id'].astype(str),
            train_trans['article_id'].astype(str)
        ))
        print(f"📊 标签期交易: {len(train_trans):,} 条, 标签集大小: {len(label_set):,}")
        del train_trans
        gc.collect()

        # 采样 20 万用户训练
        TRAIN_SAMPLE = 200000
        np.random.seed(42)
        sampled = np.random.choice(customers['customer_id'].unique(),
                                   min(TRAIN_SAMPLE, len(customers)), replace=False)
        print(f"\n🎯 采样 {len(sampled):,} 用户训练 LGBM...")

        train_df = _build_training_samples(final_recs, source_info, label_set, sampled)
        train_df = extract_advanced_features_for_twostage(train_df, transactions, customers, articles)
        print(f"训练样本: {len(train_df):,} 行, 正样本: {train_df['purchased'].sum():,} ({train_df['purchased'].mean()*100:.2f}%)")
        print_memory_usage("训练数据准备完成")

        # 训练 LGBM
        from src.ranker import train_lgbm_ranker
        FEATURES = [
            "item_popularity", "user_activity", "is_repurchase",
            "is_from_repurchase", "is_from_itemcf", "is_from_usercf", "is_from_w2v", "is_from_popularity",
            "repurchase_rank", "itemcf_rank", "usercf_rank", "w2v_rank",
            "is_category_matched", "is_color_matched",
            "item_price", "user_avg_price", "price_diff", "item_age_weeks",
            "user_price_p50", "user_price_p10", "user_price_p90",
            "user_activity_log", "user_activity_bucket",
            "is_new_repurchase",
            "user_recency_days", "user_monetary", "user_avg_basket",
            "age", "club_member_status", "fashion_news_frequency",
            "product_group_name", "index_group_name", "colour_group_name", "graphical_appearance_name",
        ]
        CAT_COLS = [
            "club_member_status", "fashion_news_frequency",
            "product_group_name", "index_group_name",
            "colour_group_name", "graphical_appearance_name",
            "user_activity_bucket",
        ]
        ranker = train_lgbm_ranker(train_df, FEATURES,
            params={'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 63},
            categorical_cols=CAT_COLS)
        del train_df
        gc.collect()

        # 分批预测全量用户
        print("\n🔮 LGBM 全量推理...")
        lookups = precompute_feature_lookups(transactions, articles)
        all_users = customers['customer_id'].unique()
        ranked_recs = {}
        BATCH = 50000
        n_batches = (len(all_users) + BATCH - 1) // BATCH
        for b in range(n_batches):
            batch = all_users[b * BATCH:(b + 1) * BATCH]
            print(f"  批次 {b + 1}/{n_batches}: 用户 {b * BATCH:,}-{min((b + 1) * BATCH, len(all_users)):,}")
            batch_source = _build_source_for(final_recs, source_info, batch)
            samples = []
            for uid in batch:
                su = str(uid)
                if su not in final_recs:
                    continue
                usrc = batch_source.get(su, {})
                for item_id in final_recs[su]:
                    src = usrc.get(str(item_id), {})
                    samples.append({
                        "customer_id": uid, "article_id": item_id, "purchased": 0,
                        "is_from_repurchase": src.get("is_from_repurchase", 0),
                        "is_from_itemcf": src.get("is_from_itemcf", 0),
                        "is_from_usercf": src.get("is_from_usercf", 0),
                        "is_from_w2v": src.get("is_from_w2v", 0),
                        "is_from_popularity": src.get("is_from_popularity", 0),
                        "repurchase_rank": src.get("repurchase_rank", 999),
                        "itemcf_rank": src.get("itemcf_rank", 999),
                        "usercf_rank": src.get("usercf_rank", 999),
                        "w2v_rank": src.get("w2v_rank", 999),
                    })
            batch_df = pd.DataFrame(samples)
            batch_df = _merge_customer_item_profiles(batch_df, customers, articles)
            batch_df = apply_feature_lookups(batch_df, lookups, articles)
            preds = ranker.predict(batch_df[FEATURES])
            batch_df["score"] = preds
            top12 = (batch_df[["customer_id", "article_id", "score"]]
                     .sort_values(["customer_id", "score"], ascending=[True, False])
                     .groupby("customer_id").head(12))
            for uid, items in top12.groupby("customer_id")["article_id"].apply(list).items():
                ranked_recs[uid] = items
            del batch_df, samples, batch_source, preds, top12
            gc.collect()
        for uid in all_users:
            if str(uid) not in ranked_recs:
                ranked_recs[str(uid)] = []
        del lookups, transactions
        gc.collect()
        generate_submission(ranked_recs, customers, uint_to_hex_cust, exp_name=exp_name)

    elif args.use_deepfm:
        # DeepFM（略，代码不变）
        from src.deepfm import DeepFM
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        candidates_df = _build_training_samples(final_recs, source_info, set(), customers['customer_id'].unique())
        candidates_df = extract_advanced_features_for_twostage(candidates_df, transactions, customers, articles)
        del transactions
        gc.collect()
        checkpoint = torch.load("models/deepfm.pt", map_location=device, weights_only=False)
        model = DeepFM(cat_feat_info=checkpoint["cat_feat_info"],
                       num_feat_names=checkpoint["num_feat_names"],
                       embed_dim=checkpoint["embed_dim"],
                       mlp_dims=checkpoint["mlp_dims"], dropout=0.0).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        from torch.utils.data import Dataset, DataLoader
        class InfDataset(Dataset):
            def __init__(self, df, cats, nums, vocabs):
                e = df.copy()
                for c in cats: e[c] = e[c].map(lambda x: vocabs[c].get(x, 0)).astype(np.int64)
                self.c = {c: torch.from_numpy(e[c].values.astype(np.int64)) for c in cats}
                self.n = torch.from_numpy(e[nums].values.astype(np.float32))
            def __len__(self): return len(self.n)
            def __getitem__(self, i): return ({c: self.c[c][i] for c in self.c}, self.n[i])
        cats = checkpoint["cat_feat_names"]
        nums = checkpoint["num_feat_names"]
        vocabs = checkpoint["cat_vocabs"]
        ds = InfDataset(candidates_df, cats, nums, vocabs)
        dl = DataLoader(ds, batch_size=8192, shuffle=False)
        all_scores = []
        with torch.no_grad():
            for cx, nx in dl:
                cx = {k: v.to(device) for k, v in cx.items()}
                all_scores.append(torch.sigmoid(model(cx, nx.to(device))).cpu().numpy())
        candidates_df["score"] = np.concatenate(all_scores)
        top12 = (candidates_df[["customer_id", "article_id", "score"]]
                 .sort_values(["customer_id", "score"], ascending=[True, False])
                 .groupby("customer_id").head(12))
        ranked_recs = top12.groupby("customer_id")["article_id"].apply(list).to_dict()
        del candidates_df, model
        gc.collect()
        generate_submission(ranked_recs, customers, uint_to_hex_cust, exp_name=exp_name)

    else:
        # 纯召回
        generate_submission(final_recs, customers, uint_to_hex_cust, exp_name=exp_name)

    print(f"\n📊 统计: {len(customers):,} 用户")
    print(f"🎉 完成！将 submissions/submission_{exp_name}.csv 提交到 Kaggle。")
