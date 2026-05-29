#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepFM 训练+评估脚本。

流程：
1. 加载 offline_data (5w 用户采样)
2. 对 train_users 做多路召回 → 构造精排样本
3. 训练 DeepFM
4. 对 valid_users 做多路召回 → 预测 → MAP@12 评估
5. 保存模型 + 类别词汇表
"""

import gc
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append("src")

from src.deepfm import DeepFM
from src.features import extract_advanced_features_for_twostage
from src.metrics import calculate_map_at_k
from src.recall_merged import RecallManager
from src.utils import print_memory_usage, reduce_mem_usage, normalize_listlike

# ========== 配置 ==========
TOP_N = 150
ITEMCF_TOP_K = 150
MAX_ITEMS = 5000
RECALL_CUTOFF = 100

EMBED_DIM = 16
MLP_DIMS = (256, 128, 64)
DROPOUT = 0.3
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 50
EARLY_STOP_PATIENCE = 5

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

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ========== 数据加载 ==========

def load_local_data():
    print("📂 加载离线数据与画像...", flush=True)
    offline_dir = "./offline_data"
    raw_dir = "./data"

    train_trans = pd.read_parquet(f"{offline_dir}/offline_train_trans.parquet")
    val_customers = pd.read_parquet(f"{offline_dir}/offline_users.parquet")
    val_ground_truth = pd.read_parquet(f"{offline_dir}/offline_val_truth.parquet")

    if "purchased_articles" not in val_ground_truth.columns:
        if "article_id" in val_ground_truth.columns:
            if not isinstance(val_ground_truth["article_id"].iloc[0], (list, np.ndarray)):
                val_ground_truth = (
                    val_ground_truth.groupby("customer_id")["article_id"]
                    .apply(list).reset_index()
                )
            val_ground_truth = val_ground_truth.rename(columns={"article_id": "purchased_articles"})

    customers = pd.read_csv(f"{raw_dir}/customers.csv")
    articles = pd.read_csv(f"{raw_dir}/articles.csv")

    # 统一 str 类型
    train_trans["customer_id"] = train_trans["customer_id"].astype(str)
    val_customers["customer_id"] = val_customers["customer_id"].astype(str)
    val_ground_truth["customer_id"] = val_ground_truth["customer_id"].astype(str)
    customers["customer_id"] = customers["customer_id"].astype(str)
    val_ground_truth["purchased_articles"] = val_ground_truth["purchased_articles"].apply(normalize_listlike)

    print("✅ 数据加载完成", flush=True)
    return train_trans, val_customers, val_ground_truth, customers, articles


def split_users(val_customers, val_ground_truth):
    all_users = val_customers["customer_id"].astype(str).to_numpy(copy=True)
    np.random.seed(42)
    np.random.shuffle(all_users)
    split_idx = int(len(all_users) * 0.8)
    train_users = set(all_users[:split_idx])
    valid_users = set(all_users[split_idx:])

    train_users_df = val_customers[val_customers["customer_id"].isin(train_users)].copy()
    valid_users_df = val_customers[val_customers["customer_id"].isin(valid_users)].copy()
    train_truth_df = val_ground_truth[val_ground_truth["customer_id"].isin(train_users)].copy()
    valid_truth_df = val_ground_truth[val_ground_truth["customer_id"].isin(valid_users)].copy()

    train_truth_dict = {str(k): normalize_listlike(v) for k, v in
                        zip(train_truth_df["customer_id"], train_truth_df["purchased_articles"])}
    valid_truth_dict = {str(k): normalize_listlike(v) for k, v in
                        zip(valid_truth_df["customer_id"], valid_truth_df["purchased_articles"])}

    print(f"🔪 用户切分: train={len(train_users)}, valid={len(valid_users)}", flush=True)
    return train_users_df, valid_users_df, train_truth_df, valid_truth_df, train_truth_dict, valid_truth_dict


def create_ranking_samples(users_df, truth_df, recall_recs, source_info):
    """从召回结果构造精排正负样本。"""
    print("📋 构造精排样本表...", flush=True)
    str_source_info = {str(u): {str(i): v for i, v in items.items()} for u, items in source_info.items()}
    ground_truth = {str(k): set(str(x) for x in normalize_listlike(v))
                    for k, v in zip(truth_df["customer_id"], truth_df["purchased_articles"])}

    samples = []
    for user_id in tqdm(users_df["customer_id"], desc="生成精排样本"):
        if user_id not in recall_recs:
            continue
        purchased_items = ground_truth.get(str(user_id), set())
        for item_id in recall_recs[user_id]:
            safe_i = str(item_id)
            source = str_source_info.get(str(user_id), {}).get(safe_i, {})
            samples.append({
                "customer_id": user_id,
                "article_id": item_id,
                "purchased": 1 if safe_i in purchased_items else 0,
                "is_from_repurchase": source.get("is_from_repurchase", 0),
                "is_from_itemcf": source.get("is_from_itemcf", 0),
                "is_from_popularity": source.get("is_from_popularity", 0),
                "repurchase_rank": source.get("repurchase_rank", 999),
                "itemcf_rank": source.get("itemcf_rank", 999),
            })
    df = pd.DataFrame(samples)
    df = reduce_mem_usage(df)
    return df


def run_recall(train_trans, users_df):
    """跑多路召回，返回 (recs, source_info)。"""
    mgr = RecallManager(
        train_trans=train_trans,
        val_customers=users_df,
        top_n=TOP_N,
        itemcf_top_k=ITEMCF_TOP_K,
        max_items=MAX_ITEMS,
        recall_cutoff=RECALL_CUTOFF,
    )
    recs, source_info = mgr.multi_recall_with_source_info()
    return recs, source_info


# ========== 类别词汇表 ==========

def build_cat_vocabs(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[str, int]]:
    """为每个类别特征构建 str→idx 映射。保留 0 给未知值。"""
    vocabs = {}
    for col in cat_cols:
        unique_vals = df[col].dropna().unique().tolist()
        vocab = {val: i + 1 for i, val in enumerate(unique_vals)}  # 0 = unknown
        vocabs[col] = vocab
        print(f"  {col}: {len(vocab)} 个类别值", flush=True)
    return vocabs


def encode_cat_df(df: pd.DataFrame, cat_cols: List[str], vocabs: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """将类别特征 str 值转为 int 索引。"""
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].map(lambda x: vocabs[col].get(x, 0)).astype(np.int64)
    return df


# ========== PyTorch Dataset ==========

class DeepFMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str], label_col: str = "purchased"):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.cat_data = {col: torch.from_numpy(df[col].values.astype(np.int64)) for col in cat_cols}
        self.num_data = torch.from_numpy(df[num_cols].values.astype(np.float32))
        self.labels = torch.from_numpy(df[label_col].values.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            {col: self.cat_data[col][idx] for col in self.cat_cols},
            self.num_data[idx],
            self.labels[idx],
        )


# ========== 训练 ==========

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for cat_x, num_x, labels in tqdm(loader, desc="训练", leave=False):
        cat_x = {k: v.to(device) for k, v in cat_x.items()}
        num_x = num_x.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(cat_x, num_x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for cat_x, num_x, labels in loader:
        cat_x = {k: v.to(device) for k, v in cat_x.items()}
        num_x = num_x.to(device)
        logits = model(cat_x, num_x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    from sklearn.metrics import roc_auc_score
    pos_ratio = y_true.mean()
    if pos_ratio == 0 or pos_ratio == 1:
        return 0.5
    return roc_auc_score(y_true, y_pred)


# ========== DeepFM 推理（全量预测→排序→MAP） ==========

def predict_and_rank(model, df, cat_cols, num_cols, vocabs, device, batch_size=8192):
    """对 df 中的每个 (user, item) 预测分数，返回每个用户 Top-12 推荐。"""
    print("🔮 DeepFM 批量预测...", flush=True)
    df_enc = encode_cat_df(df, cat_cols, vocabs)
    dataset = DeepFMDataset(df_enc, cat_cols, num_cols)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_scores = []
    with torch.no_grad():
        for cat_x, num_x, _ in tqdm(loader, desc="预测", leave=False):
            cat_x = {k: v.to(device) for k, v in cat_x.items()}
            num_x = num_x.to(device)
            logits = model(cat_x, num_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)

    df = df.copy()
    df["score"] = np.concatenate(all_scores)

    # 取每个用户 Top-12
    sorted_df = df[["customer_id", "article_id", "score"]].sort_values(
        ["customer_id", "score"], ascending=[True, False]
    )
    top12 = sorted_df.groupby("customer_id").head(12)
    ranked_recs = top12.groupby("customer_id")["article_id"].apply(list).to_dict()
    return ranked_recs


# ========== Main ==========

def main():
    print("=" * 70)
    print("🚀 DeepFM 训练 + 评估")
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟 设备: {device}", flush=True)

    # 1. 加载数据
    train_trans, val_customers, val_ground_truth, customers, articles = load_local_data()
    print_memory_usage("数据加载完成")

    train_users_df, valid_users_df, train_truth_df, valid_truth_df, train_truth_dict, valid_truth_dict = \
        split_users(val_customers, val_ground_truth)

    # 2. 跑召回
    print("\n" + "=" * 70)
    print("🎯 多路召回 (train)")
    print("=" * 70)
    train_recs, train_source = run_recall(train_trans, train_users_df)
    gc.collect()

    print("\n" + "=" * 70)
    print("🎯 多路召回 (valid)")
    print("=" * 70)
    valid_recs, valid_source = run_recall(train_trans, valid_users_df)
    gc.collect()

    # 3. 构造精排样本
    print("\n" + "=" * 70)
    print("📦 构造精排样本 + 特征提取")
    print("=" * 70)
    train_rank_df = create_ranking_samples(train_users_df, train_truth_df, train_recs, train_source)
    valid_rank_df = create_ranking_samples(valid_users_df, valid_truth_df, valid_recs, valid_source)

    full_rank_df = pd.concat([train_rank_df, valid_rank_df], axis=0, ignore_index=True)
    del train_rank_df, valid_rank_df
    gc.collect()

    full_rank_df = extract_advanced_features_for_twostage(full_rank_df, train_trans, customers, articles)
    full_rank_df = reduce_mem_usage(full_rank_df)
    print_memory_usage("特征提取完成")

    train_user_set = set(train_users_df["customer_id"].tolist())
    valid_user_set = set(valid_users_df["customer_id"].tolist())
    rank_train_df = full_rank_df[full_rank_df["customer_id"].isin(train_user_set)].copy()
    rank_valid_df = full_rank_df[full_rank_df["customer_id"].isin(valid_user_set)].copy()
    del full_rank_df
    gc.collect()

    # 4. 构建类别词汇表
    print("\n📖 构建类别特征词汇表...", flush=True)
    cat_vocabs = build_cat_vocabs(rank_train_df, CAT_FEATURES)
    cat_feat_info = {name: len(vocab) for name, vocab in cat_vocabs.items()}

    # 5. 创建 Dataset / DataLoader
    train_enc = encode_cat_df(rank_train_df, CAT_FEATURES, cat_vocabs)
    valid_enc = encode_cat_df(rank_valid_df, CAT_FEATURES, cat_vocabs)

    train_dataset = DeepFMDataset(train_enc, CAT_FEATURES, NUM_FEATURES)
    valid_dataset = DeepFMDataset(valid_enc, CAT_FEATURES, NUM_FEATURES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 6. 初始化模型
    model = DeepFM(
        cat_feat_info=cat_feat_info,
        num_feat_names=NUM_FEATURES,
        embed_dim=EMBED_DIM,
        mlp_dims=MLP_DIMS,
        dropout=DROPOUT,
    ).to(device)

    print(f"\n📐 模型参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # 7. 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_epoch = 0
    patience_counter = 0

    print("\n" + "=" * 70)
    print("🏋️ 开始训练")
    print("=" * 70, flush=True)

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        auc = evaluate_auc(model, valid_loader, device)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={loss:.4f}  valid_auc={auc:.4f}  "
              f"{'*' if auc > best_auc else ' '}", flush=True)

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            patience_counter = 0
            # 保存模型 + 词汇表
            checkpoint = {
                "model_state": model.state_dict(),
                "cat_feat_info": cat_feat_info,
                "cat_vocabs": cat_vocabs,
                "num_feat_names": NUM_FEATURES,
                "cat_feat_names": CAT_FEATURES,
                "embed_dim": EMBED_DIM,
                "mlp_dims": MLP_DIMS,
                "best_auc": best_auc,
            }
            torch.save(checkpoint, f"{MODEL_DIR}/deepfm.pt")
            print(f"   💾 模型已保存 (AUC={best_auc:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"⏹️  Early stopping at epoch {epoch} (best: epoch {best_epoch}, AUC={best_auc:.4f})")
                break

    # 8. 加载最佳模型 + 评估 MAP@12
    print("\n" + "=" * 70)
    print("📊 加载最佳模型，评估 valid_users MAP@12")
    print("=" * 70, flush=True)

    checkpoint = torch.load(f"{MODEL_DIR}/deepfm.pt", map_location=device, weights_only=False)
    model = DeepFM(
        cat_feat_info=checkpoint["cat_feat_info"],
        num_feat_names=checkpoint["num_feat_names"],
        embed_dim=checkpoint["embed_dim"],
        mlp_dims=checkpoint["mlp_dims"],
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    ranked_recs = predict_and_rank(model, rank_valid_df, CAT_FEATURES, NUM_FEATURES, cat_vocabs, device)
    for uid in valid_truth_dict.keys():
        if uid not in ranked_recs:
            ranked_recs[uid] = []

    map_score = calculate_map_at_k(valid_truth_dict, ranked_recs, k=12)
    print(f"\n🎯 DeepFM MAP@12: {map_score:.6f}", flush=True)

    # 9. 汇总
    print("\n" + "★" * 70)
    print("🏆 DeepFM 线下验证结果")
    print("★" * 70)
    print(f"  设备:           {device}")
    print(f"  训练样本:        {len(train_dataset):,}")
    print(f"  验证样本:        {len(valid_dataset):,}")
    print(f"  最佳 Valid AUC:  {best_auc:.4f}")
    print(f"  DeepFM MAP@12:  {map_score:.6f}")
    print(f"  模型路径:        {MODEL_DIR}/deepfm.pt")
    print("★" * 70, flush=True)


if __name__ == "__main__":
    main()
