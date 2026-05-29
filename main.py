#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量跑批引擎 — 多路召回 → 可选精排 → 生成 Kaggle 提交文件

策略：
- 默认：纯召回（不上 Ranker）
- --use_ranker：召回 + LGBM 精排
- --use_deepfm：召回 + DeepFM 精排
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
from src.features import extract_advanced_features_for_twostage
from src.ranker import train_lgbm_ranker
from src.utils import print_memory_usage, normalize_listlike

RANKER_FEATURE_COLS = [
    "item_popularity", "user_activity", "is_repurchase",
    "is_from_repurchase", "is_from_itemcf", "is_from_popularity",
    "repurchase_rank", "itemcf_rank",
    "is_category_matched", "is_color_matched",
    "item_price", "user_avg_price", "price_diff", "item_age_weeks",
    "user_price_p50", "user_price_p10", "user_price_p90",
    "user_activity_log", "user_activity_bucket",
    "is_new_repurchase",
    "age", "club_member_status", "fashion_news_frequency",
    "product_group_name", "index_group_name",
    "colour_group_name", "graphical_appearance_name",
]

RANKER_CAT_COLS = [
    "club_member_status", "fashion_news_frequency",
    "product_group_name", "index_group_name",
    "colour_group_name", "graphical_appearance_name",
    "user_activity_bucket",
]

DEEPFM_CAT_FEATURES = [
    "club_member_status", "fashion_news_frequency",
    "product_group_name", "index_group_name",
    "colour_group_name", "graphical_appearance_name",
]

DEEPFM_NUM_FEATURES = [
    "item_popularity", "user_activity", "is_repurchase",
    "is_from_repurchase", "is_from_itemcf", "is_from_popularity",
    "repurchase_rank", "itemcf_rank",
    "is_category_matched", "is_color_matched",
    "item_price", "user_avg_price", "price_diff", "item_age_weeks",
    "user_price_p50", "user_price_p10", "user_price_p90",
    "user_activity_log",
    "is_new_repurchase",
    "age",
]


def load_full_data(with_profiles=False):
    """加载全量数据。with_profiles=True 时额外加载画像用于特征提取。"""
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
    """将推荐结果转为 Kaggle 提交文件"""
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
            line = f"{u_hex},{' '.join(str(x).zfill(10) for x in items[:12])}\n"
            f.write(line)

    elapsed = time.time() - start
    print(f"✅ 提交文件已保存: {output_path}")
    print(f"⏱️  耗时: {elapsed:.1f} 秒")
    return output_path


def build_ranking_candidates(recs, source_info, customers, transactions=None, sample_users=None):
    """从召回结果构建候选 DataFrame，可附带标签。"""
    print("📋 构建排序候选集...", flush=True)

    purchase_set = None
    if transactions is not None:
        purchase_set = set(zip(
            transactions['customer_id'].astype(str),
            transactions['article_id'].astype(str)
        ))

    all_users = customers['customer_id'].unique()
    if sample_users is not None:
        all_users = np.random.RandomState(42).choice(all_users, min(sample_users, len(all_users)), replace=False)

    str_source_info = {str(u): {str(i): v for i, v in items.items()} for u, items in source_info.items()}
    samples = []

    for uid in all_users:
        safe_u = str(uid)
        if safe_u not in recs:
            continue
        user_source = str_source_info.get(safe_u, {})
        for item_id in recs[safe_u]:
            safe_i = str(item_id)
            src = user_source.get(safe_i, {})
            label = 1 if purchase_set and (safe_u, safe_i) in purchase_set else 0
            samples.append({
                "customer_id": uid,
                "article_id": item_id,
                "purchased": label,
                "is_from_repurchase": src.get("is_from_repurchase", 0),
                "is_from_itemcf": src.get("is_from_itemcf", 0),
                "is_from_popularity": src.get("is_from_popularity", 0),
                "repurchase_rank": src.get("repurchase_rank", 999),
                "itemcf_rank": src.get("itemcf_rank", 999),
            })

    return pd.DataFrame(samples)


def batch_predict_with_ranker(ranker, final_recs, source_info, customers, transactions, articles,
                              batch_size=50000):
    """分批预测：每次取 batch_size 个用户，构建候选→特征→预测→收集 Top-12，避免内存爆。"""
    print("🔮 LGBM 分批全量推理...", flush=True)

    all_users = customers['customer_id'].unique()
    str_source_info = {str(u): {str(i): v for i, v in items.items()} for u, items in source_info.items()}
    ranked_recs = {}

    for start in range(0, len(all_users), batch_size):
        batch = all_users[start:start + batch_size]
        print(f"  批次 {start // batch_size + 1}/{(len(all_users) + batch_size - 1) // batch_size}: "
              f"用户 {start:,} - {min(start + batch_size, len(all_users)):,}", flush=True)

        samples = []
        for uid in batch:
            safe_u = str(uid)
            if safe_u not in final_recs:
                continue
            user_source = str_source_info.get(safe_u, {})
            for item_id in final_recs[safe_u]:
                src = user_source.get(str(item_id), {})
                samples.append({
                    "customer_id": uid, "article_id": item_id, "purchased": 0,
                    "is_from_repurchase": src.get("is_from_repurchase", 0),
                    "is_from_itemcf": src.get("is_from_itemcf", 0),
                    "is_from_popularity": src.get("is_from_popularity", 0),
                    "repurchase_rank": src.get("repurchase_rank", 999),
                    "itemcf_rank": src.get("itemcf_rank", 999),
                })

        batch_df = pd.DataFrame(samples)
        batch_df = extract_advanced_features_for_twostage(batch_df, transactions, customers, articles)

        preds = ranker.predict(batch_df[RANKER_FEATURE_COLS])
        batch_df["score"] = preds

        top12 = (
            batch_df[["customer_id", "article_id", "score"]]
            .sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id")
            .head(12)
        )
        for uid, items in top12.groupby("customer_id")["article_id"].apply(list).items():
            ranked_recs[uid] = items

        del batch_df, samples, batch, preds, top12
        gc.collect()

    # 兜底：没有预测结果的用户给全局热门
    popular_items = list(final_recs[list(final_recs.keys())[0]][:12]) if final_recs else []
    for uid in all_users:
        safe_u = str(uid)
        if safe_u not in ranked_recs:
            ranked_recs[safe_u] = []

    return ranked_recs
    """用 LGBM 对候选集重排，返回每个用户 Top-12。"""
    print("🔮 LGBM 全量推理...", flush=True)
    batch_size = 500000
    preds = []
    for i in range(0, len(candidates_df), batch_size):
        preds.append(ranker.predict(candidates_df[RANKER_FEATURE_COLS].iloc[i:i + batch_size]))
    candidates_df = candidates_df.copy()
    candidates_df["score"] = np.concatenate(preds)

    top12 = (
        candidates_df[["customer_id", "article_id", "score"]]
        .sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id")
        .head(12)
    )
    return top12.groupby("customer_id")["article_id"].apply(list).to_dict()


def load_deepfm_model(model_path, device):
    """加载 DeepFM 模型和词汇表。"""
    import torch
    from src.deepfm import DeepFM

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = DeepFM(
        cat_feat_info=checkpoint["cat_feat_info"],
        num_feat_names=checkpoint["num_feat_names"],
        embed_dim=checkpoint["embed_dim"],
        mlp_dims=checkpoint["mlp_dims"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["cat_vocabs"]


def deepfm_rerank(model, cat_vocabs, df, device, batch_size=8192):
    """用 DeepFM 对候选集重排。"""
    import torch
    from torch.utils.data import Dataset, DataLoader

    class InferenceDataset(Dataset):
        def __init__(self, df, cat_cols, num_cols, vocabs):
            df_enc = df.copy()
            for col in cat_cols:
                df_enc[col] = df_enc[col].map(lambda x: vocabs[col].get(x, 0)).astype(np.int64)
            self.cat_data = {col: torch.from_numpy(df_enc[col].values.astype(np.int64)) for col in cat_cols}
            self.num_data = torch.from_numpy(df_enc[num_cols].values.astype(np.float32))

        def __len__(self):
            return len(self.num_data)

        def __getitem__(self, idx):
            return ({col: self.cat_data[col][idx] for col in cat_cols}, self.num_data[idx])

    print("🔮 DeepFM 全量推理...", flush=True)
    dataset = InferenceDataset(df, DEEPFM_CAT_FEATURES, DEEPFM_NUM_FEATURES, cat_vocabs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_scores = []
    with torch.no_grad():
        for cat_x, num_x in loader:
            cat_x = {k: v.to(device) for k, v in cat_x.items()}
            num_x = num_x.to(device)
            probs = torch.sigmoid(model(cat_x, num_x)).cpu().numpy()
            all_scores.append(probs)

    df = df.copy()
    df["score"] = np.concatenate(all_scores)
    top12 = (
        df[["customer_id", "article_id", "score"]]
        .sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id")
        .head(12)
    )
    return top12.groupby("customer_id")["article_id"].apply(list).to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='recall',
                        help='Experiment name')
    parser.add_argument('--use_ranker', action='store_true',
                        help='启用 LGBM 精排')
    parser.add_argument('--use_deepfm', action='store_true',
                        help='启用 DeepFM 精排（需先训练模型）')
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
        top_n=150, itemcf_top_k=150, max_items=5000, recall_cutoff=100,
    )
    final_recs, source_info = mgr.multi_recall_with_source_info()
    print_memory_usage("召回完成")
    del mgr
    gc.collect()

    if args.use_ranker:
        # --- LGBM 精排 ---
        TRAIN_SAMPLE = 100000  # 用 10w 用户训练
        print(f"\n🎯 采样 {TRAIN_SAMPLE:,} 用户训练 LGBM...")

        train_df = build_ranking_candidates(
            final_recs, source_info, customers,
            transactions=transactions, sample_users=TRAIN_SAMPLE
        )
        train_df = extract_advanced_features_for_twostage(train_df, transactions, customers, articles)
        print(f"训练样本: {len(train_df):,} 行, 正样本: {train_df['purchased'].sum():,}", flush=True)
        print_memory_usage("训练数据准备完成")

        ranker = train_lgbm_ranker(
            train_df, RANKER_FEATURE_COLS,
            params={'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 63},
            categorical_cols=RANKER_CAT_COLS,
        )
        del train_df
        gc.collect()

        # 分批全量预测（避免一次性 1.37 亿行爆内存）
        ranked_recs = batch_predict_with_ranker(ranker, final_recs, source_info, customers, transactions, articles)
        del ranker
        gc.collect()
        generate_submission(ranked_recs, customers, uint_to_hex_cust, exp_name=exp_name)

    elif args.use_deepfm:
        # --- DeepFM 精排 ---
        candidates_df = build_ranking_candidates(final_recs, source_info, customers)
        candidates_df = extract_advanced_features_for_twostage(candidates_df, transactions, customers, articles)
        del transactions
        gc.collect()
        print_memory_usage("候选集特征提取完成")

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, cat_vocabs = load_deepfm_model("models/deepfm.pt", device)
        ranked_recs = deepfm_rerank(model, cat_vocabs, candidates_df, device)
        del candidates_df, model
        gc.collect()
        generate_submission(ranked_recs, customers, uint_to_hex_cust, exp_name=exp_name)

    else:
        # --- 纯召回 ---
        generate_submission(final_recs, customers, uint_to_hex_cust, exp_name=exp_name)

    print(f"\n📊 统计: {len(customers):,} 用户")
    print(f"🎉 完成！将 submissions/submission_{exp_name}.csv 提交到 Kaggle。")
