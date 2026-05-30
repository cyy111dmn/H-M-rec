#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study for H&M Recommendation Project

实验项：
1. baseline MAP@12
2. recall(full) MAP@12
3. ranker(full) MAP@12
4. recall(no_repurchase) MAP@12
5. recall(no_itemcf) MAP@12
6. ranker(no_price_features) MAP@12
7. ranker(no_preference_match) MAP@12

输出：
- 控制台打印结果
- 保存到 ablation_results.csv
"""

import gc
import os
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("src")

from src.features import extract_advanced_features_for_twostage
from src.metrics import calculate_map_at_k
from src.ranker import train_lgbm_ranker
from src.recall_merged import RecallManager
from src.utils import print_memory_usage, reduce_mem_usage, normalize_listlike


# =========================
# 可调参数
# =========================
TOP_N = 150
ITEMCF_TOP_K = 150
MAX_ITEMS = 5000
RECALL_CUTOFF = 100

LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "map",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "n_estimators": 300,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

ALL_FEATURE_COLS = [
    "item_popularity", "user_activity", "is_repurchase",
    "is_from_repurchase", "is_from_itemcf", "is_from_popularity",
    "repurchase_rank", "itemcf_rank",
    "is_category_matched", "is_color_matched",
    "item_price", "user_avg_price", "price_diff", "item_age_weeks",
    "user_price_p50", "user_price_p10", "user_price_p90",
    "user_activity_log", "user_activity_bucket",
    "is_new_repurchase",
    "user_recency_days", "user_monetary", "user_avg_basket",
    "age", "club_member_status", "fashion_news_frequency",
    "product_group_name", "index_group_name", "colour_group_name", "graphical_appearance_name"
]

CAT_FEATURE_COLS = [
    "club_member_status", "fashion_news_frequency",
    "product_group_name", "index_group_name",
    "colour_group_name", "graphical_appearance_name",
    "user_activity_bucket",
]

PRICE_FEATURES = ["item_price", "user_avg_price", "price_diff", "user_monetary", "user_avg_basket"]
PREFERENCE_MATCH_FEATURES = ["is_category_matched", "is_color_matched"]


# =========================
# 工具函数（委托 utils）
# =========================
# print_memory_usage, reduce_mem_usage, normalize_listlike 已迁移到 src.utils


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
                    .apply(list)
                    .reset_index()
                )
            val_ground_truth = val_ground_truth.rename(columns={"article_id": "purchased_articles"})

    customers = pd.read_csv(f"{raw_dir}/customers.csv")
    articles = pd.read_csv(f"{raw_dir}/articles.csv")

    # customer_id 全部统一成 str，和当前 recall_merged.py 对齐
    train_trans["customer_id"] = train_trans["customer_id"].astype(str)
    val_customers["customer_id"] = val_customers["customer_id"].astype(str)
    val_ground_truth["customer_id"] = val_ground_truth["customer_id"].astype(str)
    customers["customer_id"] = customers["customer_id"].astype(str)

    # truth 列统一成 list
    val_ground_truth["purchased_articles"] = val_ground_truth["purchased_articles"].apply(normalize_listlike)

    print("✅ 数据加载完成", flush=True)
    return train_trans, val_customers, val_ground_truth, customers, articles


def split_users(val_customers: pd.DataFrame, val_ground_truth: pd.DataFrame):
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

    train_truth_dict = {
        str(k): normalize_listlike(v)
        for k, v in zip(train_truth_df["customer_id"], train_truth_df["purchased_articles"])
    }
    valid_truth_dict = {
        str(k): normalize_listlike(v)
        for k, v in zip(valid_truth_df["customer_id"], valid_truth_df["purchased_articles"])
    }

    print(f"🔪 用户切分完成: train_users={len(train_users)}, valid_users={len(valid_users)}", flush=True)
    return train_users_df, valid_users_df, train_truth_df, valid_truth_df, train_truth_dict, valid_truth_dict


def build_global_popular_items(train_trans: pd.DataFrame, topk: int = 100) -> List:
    tmp = train_trans.copy()
    tmp["t_dat"] = pd.to_datetime(tmp["t_dat"])
    max_date = tmp["t_dat"].max()
    tmp["week"] = (max_date - tmp["t_dat"]).dt.days // 7

    last_week = tmp["week"].min()
    popular_items = (
        tmp[tmp["week"] == last_week]
        .groupby("article_id")
        .size()
        .sort_values(ascending=False)
        .head(topk)
        .index.tolist()
    )
    return popular_items


def run_popularity_baseline(
    train_trans: pd.DataFrame,
    eval_users: List[str],
    eval_truth_dict: Dict[str, List]
) -> float:
    popular_items = build_global_popular_items(train_trans, topk=50)
    predicted_dict = {uid: popular_items[:12] for uid in eval_users}
    return calculate_map_at_k(eval_truth_dict, predicted_dict, k=12)


def run_recall(
    train_trans: pd.DataFrame,
    eval_users_df: pd.DataFrame,
    eval_truth_dict: Dict[str, List]
) -> Tuple[float, Dict[str, List], Dict[str, Dict]]:
    recall_manager = RecallManager(
        train_trans=train_trans,
        val_customers=eval_users_df,
        top_n=TOP_N,
        itemcf_top_k=ITEMCF_TOP_K,
        max_items=MAX_ITEMS,
        recall_cutoff=RECALL_CUTOFF,
    )
    final_recs, source_info = recall_manager.multi_recall_with_source_info()
    map_score = calculate_map_at_k(eval_truth_dict, final_recs, k=12)
    return map_score, final_recs, source_info


def create_ranking_samples(
    users_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    recall_recs: Dict[str, List],
    source_info: Dict[str, Dict]
) -> pd.DataFrame:
    print("\n📋 构造精排样本表...", flush=True)

    str_source_info = {
        str(u): {str(i): v for i, v in items.items()}
        for u, items in source_info.items()
    }
    ground_truth = {
        str(k): set([str(x) for x in normalize_listlike(v)])
        for k, v in zip(truth_df["customer_id"], truth_df["purchased_articles"])
    }

    samples = []
    for user_id in tqdm(users_df["customer_id"], desc="生成精排正负样本"):
        if user_id not in recall_recs:
            continue

        safe_u = str(user_id)
        purchased_items = ground_truth.get(safe_u, set())

        for item_id in recall_recs[user_id]:
            safe_i = str(item_id)
            source = str_source_info.get(safe_u, {}).get(safe_i, {})

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


def ensure_feature_columns(df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    df = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            if col in cat_cols:
                df[col] = "UNKNOWN"
            else:
                df[col] = 0

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype("category")

    return df


def evaluate_ranker(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    valid_truth_dict: Dict[str, List],
    feature_cols: List[str],
    cat_cols: List[str],
    exp_name: str,
) -> float:
    print(f"\n🚀 训练 Ranker: {exp_name}", flush=True)

    train_df = ensure_feature_columns(train_df, feature_cols, cat_cols)
    valid_df = ensure_feature_columns(valid_df, feature_cols, cat_cols)

    ranker = train_lgbm_ranker(
        train_df=train_df,
        feature_cols=feature_cols,
        params=LGBM_PARAMS,
        categorical_cols=cat_cols,
    )

    batch_size = 500000
    preds = []
    for i in range(0, len(valid_df), batch_size):
        preds.append(ranker.predict(valid_df[feature_cols].iloc[i:i + batch_size]))

    valid_df = valid_df.copy()
    valid_df["score"] = np.concatenate(preds)

    sorted_df = valid_df[["customer_id", "article_id", "score"]].sort_values(
        ["customer_id", "score"], ascending=[True, False]
    )
    top_items_df = sorted_df.groupby("customer_id").head(12)
    ranked_recs = top_items_df.groupby("customer_id")["article_id"].apply(list).to_dict()

    for uid in valid_truth_dict.keys():
        if uid not in ranked_recs:
            ranked_recs[uid] = []

    map_score = calculate_map_at_k(valid_truth_dict, ranked_recs, k=12)
    print(f"🎯 {exp_name} MAP@12: {map_score:.6f}", flush=True)

    del ranker, valid_df, sorted_df, top_items_df, ranked_recs
    gc.collect()

    return map_score


def filter_recall_by_source(
    full_recs: Dict[str, List],
    source_info: Dict[str, Dict],
    fallback_popular_items: List,
    drop_repurchase: bool = False,
    drop_itemcf: bool = False,
    fill_limit: int = RECALL_CUTOFF,
) -> Dict[str, List]:
    """
    基于 full recall + source_info 做召回消融，不重复跑 ItemCF。
    """
    ablated_recs = {}

    for uid, items in full_recs.items():
        kept = []
        seen = set()
        user_source = source_info.get(uid, {})

        for item in items:
            info = user_source.get(item, {})
            keep = False

            if info.get("is_from_popularity", 0) == 1:
                keep = True
            if not drop_repurchase and info.get("is_from_repurchase", 0) == 1:
                keep = True
            if not drop_itemcf and info.get("is_from_itemcf", 0) == 1:
                keep = True

            if keep and item not in seen:
                kept.append(item)
                seen.add(item)

        if len(kept) < fill_limit:
            for pop_item in fallback_popular_items:
                if pop_item not in seen:
                    kept.append(pop_item)
                    seen.add(pop_item)
                if len(kept) >= fill_limit:
                    break

        ablated_recs[uid] = kept[:fill_limit]

    return ablated_recs


def main():
    print("🚀 启动 Ablation Study", flush=True)
    train_trans, val_customers, val_ground_truth, customers, articles = load_local_data()
    print_memory_usage("数据加载完成")

    (
        train_users_df,
        valid_users_df,
        train_truth_df,
        valid_truth_df,
        train_truth_dict,
        valid_truth_dict,
    ) = split_users(val_customers, val_ground_truth)

    # 1) Baseline
    print("\n" + "=" * 70)
    print("🔥 Baseline")
    print("=" * 70)
    baseline_map = run_popularity_baseline(
        train_trans=train_trans,
        eval_users=valid_users_df["customer_id"].tolist(),
        eval_truth_dict=valid_truth_dict,
    )
    print(f"🎯 baseline MAP@12: {baseline_map:.6f}", flush=True)

    # 2) Recall on train users / valid users
    print("\n" + "=" * 70)
    print("🎯 Full Recall on train_users")
    print("=" * 70)
    _, train_full_recs, train_source_info = run_recall(
        train_trans=train_trans,
        eval_users_df=train_users_df,
        eval_truth_dict=train_truth_dict,
    )

    print("\n" + "=" * 70)
    print("🎯 Full Recall on valid_users")
    print("=" * 70)
    recall_full_map, valid_full_recs, valid_source_info = run_recall(
        train_trans=train_trans,
        eval_users_df=valid_users_df,
        eval_truth_dict=valid_truth_dict,
    )
    print(f"🎯 recall(full) MAP@12: {recall_full_map:.6f}", flush=True)

    # 3) Recall ablation
    fallback_popular_items = build_global_popular_items(train_trans, topk=RECALL_CUTOFF)

    valid_no_repurchase_recs = filter_recall_by_source(
        full_recs=valid_full_recs,
        source_info=valid_source_info,
        fallback_popular_items=fallback_popular_items,
        drop_repurchase=True,
        drop_itemcf=False,
        fill_limit=RECALL_CUTOFF,
    )
    recall_no_repurchase_map = calculate_map_at_k(valid_truth_dict, valid_no_repurchase_recs, k=12)
    print(f"🎯 recall(no_repurchase) MAP@12: {recall_no_repurchase_map:.6f}", flush=True)

    valid_no_itemcf_recs = filter_recall_by_source(
        full_recs=valid_full_recs,
        source_info=valid_source_info,
        fallback_popular_items=fallback_popular_items,
        drop_repurchase=False,
        drop_itemcf=True,
        fill_limit=RECALL_CUTOFF,
    )
    recall_no_itemcf_map = calculate_map_at_k(valid_truth_dict, valid_no_itemcf_recs, k=12)
    print(f"🎯 recall(no_itemcf) MAP@12: {recall_no_itemcf_map:.6f}", flush=True)

    # 4) Build ranking samples once
    print("\n" + "=" * 70)
    print("📦 构造 Ranker 样本并一次性提特征")
    print("=" * 70)
    train_rank_df = create_ranking_samples(
        users_df=train_users_df,
        truth_df=train_truth_df,
        recall_recs=train_full_recs,
        source_info=train_source_info,
    )
    valid_rank_df = create_ranking_samples(
        users_df=valid_users_df,
        truth_df=valid_truth_df,
        recall_recs=valid_full_recs,
        source_info=valid_source_info,
    )

    full_rank_df = pd.concat([train_rank_df, valid_rank_df], axis=0, ignore_index=True)
    del train_rank_df, valid_rank_df
    gc.collect()

    full_rank_df = extract_advanced_features_for_twostage(
        full_rank_df,
        train_trans,
        customers,
        articles,
    )
    full_rank_df = reduce_mem_usage(full_rank_df)
    print_memory_usage("特征提取完成")

    train_user_set = set(train_users_df["customer_id"].tolist())
    valid_user_set = set(valid_users_df["customer_id"].tolist())

    rank_train_df = full_rank_df[full_rank_df["customer_id"].isin(train_user_set)].copy()
    rank_valid_df = full_rank_df[full_rank_df["customer_id"].isin(valid_user_set)].copy()

    del full_rank_df
    gc.collect()

    # 5) Ranker(full)
    ranker_full_map = evaluate_ranker(
        train_df=rank_train_df,
        valid_df=rank_valid_df,
        valid_truth_dict=valid_truth_dict,
        feature_cols=ALL_FEATURE_COLS,
        cat_cols=CAT_FEATURE_COLS,
        exp_name="ranker(full)",
    )

    # 6) Ranker(no_price_features)
    feature_cols_no_price = [c for c in ALL_FEATURE_COLS if c not in PRICE_FEATURES]
    cat_cols_no_price = [c for c in CAT_FEATURE_COLS if c in feature_cols_no_price]
    ranker_no_price_map = evaluate_ranker(
        train_df=rank_train_df,
        valid_df=rank_valid_df,
        valid_truth_dict=valid_truth_dict,
        feature_cols=feature_cols_no_price,
        cat_cols=cat_cols_no_price,
        exp_name="ranker(no_price_features)",
    )

    # 7) Ranker(no_preference_match)
    feature_cols_no_pref = [c for c in ALL_FEATURE_COLS if c not in PREFERENCE_MATCH_FEATURES]
    cat_cols_no_pref = [c for c in CAT_FEATURE_COLS if c in feature_cols_no_pref]
    ranker_no_pref_map = evaluate_ranker(
        train_df=rank_train_df,
        valid_df=rank_valid_df,
        valid_truth_dict=valid_truth_dict,
        feature_cols=feature_cols_no_pref,
        cat_cols=cat_cols_no_pref,
        exp_name="ranker(no_preference_match)",
    )

    # 8) 汇总结果
    results = pd.DataFrame([
        {"experiment": "baseline", "map_at_12": baseline_map},
        {"experiment": "recall(full)", "map_at_12": recall_full_map},
        {"experiment": "ranker(full)", "map_at_12": ranker_full_map},
        {"experiment": "recall(no_repurchase)", "map_at_12": recall_no_repurchase_map},
        {"experiment": "recall(no_itemcf)", "map_at_12": recall_no_itemcf_map},
        {"experiment": "ranker(no_price_features)", "map_at_12": ranker_no_price_map},
        {"experiment": "ranker(no_preference_match)", "map_at_12": ranker_no_pref_map},
    ])

    results["delta_vs_full_recall"] = results["map_at_12"] - recall_full_map
    results["delta_vs_full_ranker"] = results["map_at_12"] - ranker_full_map
    results.to_csv("ablation_results.csv", index=False)

    print("\n" + "★" * 70)
    print("🏆 Ablation Study Results")
    print("★" * 70)
    print(results.to_string(index=False))
    print("\n✅ 结果已保存到 ablation_results.csv", flush=True)


# =========================
# 网格搜索入口
# =========================

def generate_weight_grid() -> List[Dict]:
    """生成融合权重网格（3 个来源权重 + 3 个 rank 权重）。"""
    repurchase_weights = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    itemcf_weights = [1.0, 1.5, 2.0, 2.5, 3.0]
    popularity_weights = [0.5, 0.8, 1.0, 1.5]

    # rank 权重保持当前值，先只调 source 权重
    grid = []
    for rw in repurchase_weights:
        for iw in itemcf_weights:
            for pw in popularity_weights:
                grid.append({
                    "repurchase_weight": rw,
                    "itemcf_weight": iw,
                    "popularity_weight": pw,
                })
    return grid


def run_grid_search():
    """加载 5w 数据 → 跑一次召回 → 网格搜索最佳融合权重。"""
    print("=" * 70)
    print("🚀 融合权重网格搜索")
    print("=" * 70)

    train_trans, val_customers, val_ground_truth, _, _ = load_local_data()

    all_users = val_customers["customer_id"].astype(str).to_numpy(copy=True)
    np.random.seed(42)
    np.random.shuffle(all_users)
    split_idx = int(len(all_users) * 0.8)
    valid_users = set(all_users[split_idx:])
    valid_users_df = val_customers[val_customers["customer_id"].isin(valid_users)].copy()
    valid_truth_df = val_ground_truth[val_ground_truth["customer_id"].isin(valid_users)].copy()
    valid_truth_dict = {
        str(k): normalize_listlike(v)
        for k, v in zip(valid_truth_df["customer_id"], valid_truth_df["purchased_articles"])
    }
    print(f"👥 验证用户: {len(valid_users)}", flush=True)

    # 创建 RecallManager（用默认权重，只做召回）
    mgr = RecallManager(
        train_trans=train_trans,
        val_customers=valid_users_df,
        top_n=TOP_N,
        itemcf_top_k=ITEMCF_TOP_K,
        max_items=MAX_ITEMS,
        recall_cutoff=RECALL_CUTOFF,
    )

    # 生成网格 + 搜索
    grid = generate_weight_grid()
    print(f"📋 网格大小: {len(grid)} 组", flush=True)

    results = mgr.grid_search_weights(valid_truth_dict, grid, k=12)

    # 排序找最优
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("map_at_k", ascending=False).reset_index(drop=True)

    print("\n" + "★" * 70)
    print("🏆 网格搜索 Top-10 权重组合")
    print("★" * 70)
    print(results_df.head(10).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n🥇 最佳权重: repurchase={best['repurchase_weight']:.1f}, "
          f"itemcf={best['itemcf_weight']:.1f}, popularity={best['popularity_weight']:.1f}")
    print(f"   MAP@12 = {best['map_at_k']:.6f}")

    # 与默认权重对比
    default_row = results_df[
        (results_df["repurchase_weight"] == 3.0) &
        (results_df["itemcf_weight"] == 1.5) &
        (results_df["popularity_weight"] == 0.8)
    ]
    if len(default_row) > 0:
        default_map = default_row.iloc[0]["map_at_k"]
        improvement = (best["map_at_k"] - default_map) / (default_map + 1e-9) * 100
        print(f"\n📊 默认权重 (3.0/1.5/0.8) MAP@12 = {default_map:.6f}")
        print(f"📈 最佳权重相对提升: {improvement:+.2f}%")

    # 保存结果
    results_df.to_csv("grid_search_weights.csv", index=False)
    print("\n✅ 结果已保存到 grid_search_weights.csv", flush=True)

    # 输出可直接复制到代码的参数字符串
    print(f"\n📝 复制到 recall_merged.py RecallManager.__init__ 默认参数:")
    print(f"    repurchase_weight={best['repurchase_weight']}, "
          f"itemcf_weight={best['itemcf_weight']}, "
          f"popularity_weight={best['popularity_weight']},")


if __name__ == "__main__":
    import sys

    if "--grid-search" in sys.argv:
        run_grid_search()
    else:
        main()