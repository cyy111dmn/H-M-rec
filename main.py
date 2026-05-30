#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量跑批引擎 — 多路召回 → 可选 DeepFM 精排 → 生成 Kaggle 提交文件

策略：
- 默认：纯召回（不上 Ranker）
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
from src.metrics import calculate_map_at_k
from src.utils import print_memory_usage, normalize_listlike

CAT_FEATURES = [
    "club_member_status", "fashion_news_frequency",
    "product_group_name", "index_group_name",
    "colour_group_name", "graphical_appearance_name",
]

NUM_FEATURES = [
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


def load_full_data(use_deepfm=False):
    """加载全量数据。use_deepfm=True 时额外加载画像。"""
    print("📂 加载全量数据...", flush=True)
    data_dir = './data'

    transactions = pd.read_csv(f'{data_dir}/transactions_train.csv')
    uint_to_hex_cust = load_id_mapping(data_dir)

    if use_deepfm:
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


def build_ranking_candidates(recs, source_info, customers):
    """从召回结果构建 (用户, 商品) 候选 DataFrame，用于特征提取和打分。"""
    print("📋 构建全量排序候选集...", flush=True)
    samples = []
    all_users = customers['customer_id'].unique()

    str_source_info = {str(u): {str(i): v for i, v in items.items()} for u, items in source_info.items()}

    for uid in all_users:
        safe_u = str(uid)
        if safe_u not in recs:
            continue
        user_source = str_source_info.get(safe_u, {})
        for item_id in recs[safe_u]:
            safe_i = str(item_id)
            src = user_source.get(safe_i, {})
            samples.append({
                "customer_id": uid,
                "article_id": item_id,
                "purchased": 0,  # 全量推理没有标签
                "is_from_repurchase": src.get("is_from_repurchase", 0),
                "is_from_itemcf": src.get("is_from_itemcf", 0),
                "is_from_popularity": src.get("is_from_popularity", 0),
                "repurchase_rank": src.get("repurchase_rank", 999),
                "itemcf_rank": src.get("itemcf_rank", 999),
            })

    return pd.DataFrame(samples)


def load_deepfm_model(model_path, device):
    """加载训练好的 DeepFM 模型和词汇表。"""
    import torch
    from src.deepfm import DeepFM

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = DeepFM(
        cat_feat_info=checkpoint["cat_feat_info"],
        num_feat_names=checkpoint["num_feat_names"],
        embed_dim=checkpoint["embed_dim"],
        mlp_dims=checkpoint["mlp_dims"],
        dropout=0.0,  # 推理时 dropout=0
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["cat_vocabs"]


def deepfm_rerank(model, cat_vocabs, df, device, batch_size=8192):
    """用 DeepFM 对候选集重排，返回每个用户 Top-12。"""
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
    dataset = InferenceDataset(df, CAT_FEATURES, NUM_FEATURES, cat_vocabs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_scores = []
    with torch.no_grad():
        for cat_x, num_x in loader:
            cat_x = {k: v.to(device) for k, v in cat_x.items()}
            num_x = num_x.to(device)
            logits = model(cat_x, num_x)
            probs = torch.sigmoid(logits).cpu().numpy()
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
                        help='Experiment name, used in output filename')
    parser.add_argument('--use_deepfm', action='store_true',
                        help='启用 DeepFM 精排（默认仅召回）')
    args = parser.parse_args()
    exp_name = args.exp_name

    mode = "召回+DeepFM精排" if args.use_deepfm else "纯召回"
    print("=" * 60)
    print(f"🚀 全量跑批 — {mode} ({exp_name})")
    print("=" * 60)

    # 1. 加载数据
    transactions, customers, articles, uint_to_hex_cust = load_full_data(use_deepfm=args.use_deepfm)
    print_memory_usage("数据加载完成")

    # 2. 多路召回（始终带来源信息，DeepFM 需要）
    print("\n--- 执行多路召回 ---")
    mgr = RecallManager(
        train_trans=transactions,
        val_customers=customers,
        top_n=150,
        itemcf_top_k=150,
        max_items=5000,
        recall_cutoff=100,
    )
    final_recs, source_info = mgr.multi_recall_with_source_info()
    print_memory_usage("召回完成")

    del mgr
    gc.collect()

    if args.use_deepfm:
        # 3a. 构造候选集 + 特征提取
        candidates_df = build_ranking_candidates(final_recs, source_info, customers)
        print(f"候选集大小: {len(candidates_df):,} 行", flush=True)
        print_memory_usage("候选集构建完成")

        # 提取特征
        candidates_df = extract_advanced_features_for_twostage(
            candidates_df, transactions, customers, articles
        )
        print_memory_usage("特征提取完成")

        del transactions
        gc.collect()

        # 4a. 加载 DeepFM + 重排
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📟 设备: {device}", flush=True)

        model, cat_vocabs = load_deepfm_model("models/deepfm.pt", device)
        ranked_recs = deepfm_rerank(model, cat_vocabs, candidates_df, device)

        del candidates_df, model
        gc.collect()

        # 5a. 生成提交文件
        generate_submission(ranked_recs, customers, uint_to_hex_cust, exp_name=exp_name)

    else:
        # 3b. 纯召回 → 提交文件
        generate_submission(final_recs, customers, uint_to_hex_cust, exp_name=exp_name)

    # 6. 统计
    print(f"\n📊 统计: {len(customers):,} 用户")
    print(f"🎉 完成！将 submissions/submission_{exp_name}.csv 提交到 Kaggle。")
