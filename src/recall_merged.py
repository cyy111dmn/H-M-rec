#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src.utils import print_memory_usage


class RecallManager:
    """多路召回管理器。

    关键改动：
    1. 仍然保留原有关键字参数接口，避免参数错位。
    2. 将“优先级拼接”改为“来源感知的加权融合”。
    3. 融合分数综合考虑：
       - 是否来自复购 / ItemCF / 热门
       - 在各路召回中的 rank
    """

    def __init__(
        self,
        train_trans: pd.DataFrame,
        val_customers: pd.DataFrame,
        *,
        top_n: int = 50,
        itemcf_top_k: int = 20,
        use_sparse_matrix: bool = True,
        max_items: int | None = None,
        recall_cutoff: int = 50,
        recent_weeks: int = 27,
        # ===== 新增：融合权重 =====
        repurchase_weight: float = 3.0,
        itemcf_weight: float = 1.5,
        popularity_weight: float = 0.8,
        repurchase_rank_weight: float = 1.0,
        itemcf_rank_weight: float = 0.5,
        popularity_rank_weight: float = 0.2,
    ):
        if not isinstance(use_sparse_matrix, bool):
            raise TypeError(
                "use_sparse_matrix 必须是 bool。请使用关键字参数调用 RecallManager，"
                "例如 RecallManager(..., max_items=5000, recall_cutoff=100)。"
            )
        if top_n <= 0 or itemcf_top_k <= 0 or recall_cutoff <= 0:
            raise ValueError("top_n / itemcf_top_k / recall_cutoff 必须为正整数")
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items 必须为正整数或 None")

        self.train_trans = train_trans.copy()
        self.val_customers = val_customers.copy()
        self.top_n = int(top_n)
        self.itemcf_top_k = int(itemcf_top_k)
        self.use_sparse_matrix = use_sparse_matrix
        self.max_items = max_items
        self.recall_cutoff = int(recall_cutoff)
        self.recent_weeks = int(recent_weeks)

        # 融合权重
        self.repurchase_weight = float(repurchase_weight)
        self.itemcf_weight = float(itemcf_weight)
        self.popularity_weight = float(popularity_weight)
        self.repurchase_rank_weight = float(repurchase_rank_weight)
        self.itemcf_rank_weight = float(itemcf_rank_weight)
        self.popularity_rank_weight = float(popularity_rank_weight)

        self._preprocess_data()

        self.popularity_recaller = PopularityRecall(self.train_trans, self.top_n)
        self.repurchase_recaller = RepurchaseRecall(self.train_trans)
        if self.use_sparse_matrix:
            self.itemcf_recaller = ItemCFRecallSparse(self.train_trans, self.itemcf_top_k)
        else:
            self.itemcf_recaller = ItemCFRecallDict(self.train_trans, self.itemcf_top_k)

    def _preprocess_data(self):
        print("预处理数据...", flush=True)
        print_memory_usage("预处理开始")

        self.train_trans["t_dat"] = pd.to_datetime(self.train_trans["t_dat"])
        max_date = self.train_trans["t_dat"].max()
        self.train_trans["week"] = (max_date - self.train_trans["t_dat"]).dt.days // 7
        self.train_trans = self.train_trans[self.train_trans["week"] < self.recent_weeks].copy()

        self.val_users = self.val_customers["customer_id"].astype(str).unique().tolist()
        self.train_trans["customer_id"] = self.train_trans["customer_id"].astype(str)

        item_counts = self.train_trans.groupby("article_id").size()
        if self.max_items is not None:
            top_items = item_counts.nlargest(self.max_items).index
            self.train_trans = self.train_trans[self.train_trans["article_id"].isin(top_items)].copy()
            print(f"商品截断：保留频次最高的 {self.max_items} 个商品", flush=True)
        else:
            print("商品截断：保留所有商品", flush=True)

        print_memory_usage("预处理完成")

    def multi_recall(self) -> Dict[str, List]:
        print("开始多路召回...", flush=True)
        popularity_recs = self.popularity_recaller.recall(self.val_users)
        repurchase_recs = self.repurchase_recaller.recall(self.val_users)
        itemcf_recs = self.itemcf_recaller.recall(self.val_users)
        final_recs = self._merge_recall_results(popularity_recs, repurchase_recs, itemcf_recs)
        print_memory_usage("多路召回完成")
        return final_recs

    def multi_recall_with_source_info(self) -> Tuple[Dict[str, List], Dict[str, Dict]]:
        print("开始多路召回（带来源信息）...", flush=True)
        popularity_recs = self.popularity_recaller.recall_with_ranks(self.val_users)
        repurchase_recs = self.repurchase_recaller.recall_with_ranks(self.val_users)
        itemcf_recs = self.itemcf_recaller.recall_with_ranks(self.val_users)
        final_recs, source_info = self._merge_recall_results_with_source(
            popularity_recs, repurchase_recs, itemcf_recs
        )
        print_memory_usage("多路召回完成")
        return final_recs, source_info

    def _safe_inv_rank(self, rank: int | None) -> float:
        if rank is None or rank <= 0:
            return 0.0
        return 1.0 / float(rank)

    def _calc_fusion_score(self, info: Dict) -> float:
        score = 0.0

        if info.get("is_from_repurchase", 0) == 1:
            score += self.repurchase_weight
            score += self.repurchase_rank_weight * self._safe_inv_rank(info.get("repurchase_rank"))

        if info.get("is_from_itemcf", 0) == 1:
            score += self.itemcf_weight
            score += self.itemcf_rank_weight * self._safe_inv_rank(info.get("itemcf_rank"))

        if info.get("is_from_popularity", 0) == 1:
            score += self.popularity_weight
            score += self.popularity_rank_weight * self._safe_inv_rank(info.get("popularity_rank"))

        return score

    def _build_user_source_info(
        self,
        user_id: str,
        popularity_recs: Dict[str, List],
        repurchase_recs: Dict[str, List],
        itemcf_recs: Dict[str, List],
    ) -> Dict:
        """构造单用户的 item -> source_info 映射。"""
        user_source_info = {}

        # 复购
        for rank, item in enumerate(repurchase_recs.get(user_id, []), start=1):
            if item not in user_source_info:
                user_source_info[item] = {
                    "is_from_repurchase": 0,
                    "is_from_itemcf": 0,
                    "is_from_popularity": 0,
                    "repurchase_rank": 999,
                    "itemcf_rank": 999,
                    "popularity_rank": 999,
                }
            user_source_info[item]["is_from_repurchase"] = 1
            user_source_info[item]["repurchase_rank"] = rank

        # ItemCF
        for rank, item in enumerate(itemcf_recs.get(user_id, []), start=1):
            if item not in user_source_info:
                user_source_info[item] = {
                    "is_from_repurchase": 0,
                    "is_from_itemcf": 0,
                    "is_from_popularity": 0,
                    "repurchase_rank": 999,
                    "itemcf_rank": 999,
                    "popularity_rank": 999,
                }
            user_source_info[item]["is_from_itemcf"] = 1
            user_source_info[item]["itemcf_rank"] = rank

        # 热门
        for rank, item in enumerate(popularity_recs.get(user_id, []), start=1):
            if item not in user_source_info:
                user_source_info[item] = {
                    "is_from_repurchase": 0,
                    "is_from_itemcf": 0,
                    "is_from_popularity": 0,
                    "repurchase_rank": 999,
                    "itemcf_rank": 999,
                    "popularity_rank": 999,
                }
            user_source_info[item]["is_from_popularity"] = 1
            user_source_info[item]["popularity_rank"] = rank

        return user_source_info

    def _rank_candidates_by_fusion(self, user_source_info: Dict) -> List:
        scored_items = []
        for item, info in user_source_info.items():
            fusion_score = self._calc_fusion_score(info)
            scored_items.append((item, fusion_score, info))

        # 分数降序；分数相同时优先更强的来源和更靠前的 rank
        scored_items.sort(
            key=lambda x: (
                -x[1],
                -x[2].get("is_from_repurchase", 0),
                -x[2].get("is_from_itemcf", 0),
                -x[2].get("is_from_popularity", 0),
                x[2].get("repurchase_rank", 999),
                x[2].get("itemcf_rank", 999),
                x[2].get("popularity_rank", 999),
            )
        )
        return [item for item, _, _ in scored_items[: self.recall_cutoff]]

    def _merge_recall_results(
        self,
        popularity_recs: Dict[str, List],
        repurchase_recs: Dict[str, List],
        itemcf_recs: Dict[str, List],
    ) -> Dict[str, List]:
        """加权融合版本，不再使用硬优先级拼接。"""
        final_recs = {}

        for user_id in tqdm(self.val_users, desc="融合召回结果", unit="用户"):
            user_source_info = self._build_user_source_info(
                user_id=user_id,
                popularity_recs=popularity_recs,
                repurchase_recs=repurchase_recs,
                itemcf_recs=itemcf_recs,
            )
            final_recs[user_id] = self._rank_candidates_by_fusion(user_source_info)

        return final_recs

    def _merge_recall_results_with_source(
        self,
        popularity_recs: Dict[str, List],
        repurchase_recs: Dict[str, List],
        itemcf_recs: Dict[str, List],
    ) -> Tuple[Dict[str, List], Dict[str, Dict]]:
        """加权融合版本，并返回最终保留候选的来源信息。"""
        final_recs = {}
        source_info = {}

        for user_id in tqdm(self.val_users, desc="融合召回结果", unit="用户"):
            user_source_info = self._build_user_source_info(
                user_id=user_id,
                popularity_recs=popularity_recs,
                repurchase_recs=repurchase_recs,
                itemcf_recs=itemcf_recs,
            )
            ranked_items = self._rank_candidates_by_fusion(user_source_info)

            final_recs[user_id] = ranked_items
            source_info[user_id] = {item: user_source_info[item] for item in ranked_items}

        return final_recs, source_info

    def _build_all_user_source_info(
        self,
        popularity_recs: Dict[str, List],
        repurchase_recs: Dict[str, List],
        itemcf_recs: Dict[str, List],
    ) -> Dict[str, Dict]:
        """为所有用户构建 source_info（一次，供网格搜索复用）。"""
        all_info = {}
        for user_id in tqdm(self.val_users, desc="构建来源信息", unit="用户"):
            all_info[user_id] = self._build_user_source_info(
                user_id=user_id,
                popularity_recs=popularity_recs,
                repurchase_recs=repurchase_recs,
                itemcf_recs=itemcf_recs,
            )
        return all_info

    def _rerank_with_weights(
        self,
        all_user_source_info: Dict[str, Dict],
        *,
        repurchase_weight: float | None = None,
        itemcf_weight: float | None = None,
        popularity_weight: float | None = None,
        repurchase_rank_weight: float | None = None,
        itemcf_rank_weight: float | None = None,
        popularity_rank_weight: float | None = None,
    ) -> Dict[str, List]:
        """在已构建的 source_info 上换权重重新排序（不重新跑召回）。"""
        # 暂存旧权重
        old = {
            "repurchase_weight": self.repurchase_weight,
            "itemcf_weight": self.itemcf_weight,
            "popularity_weight": self.popularity_weight,
            "repurchase_rank_weight": self.repurchase_rank_weight,
            "itemcf_rank_weight": self.itemcf_rank_weight,
            "popularity_rank_weight": self.popularity_rank_weight,
        }
        # 设置新权重
        if repurchase_weight is not None:
            self.repurchase_weight = repurchase_weight
        if itemcf_weight is not None:
            self.itemcf_weight = itemcf_weight
        if popularity_weight is not None:
            self.popularity_weight = popularity_weight
        if repurchase_rank_weight is not None:
            self.repurchase_rank_weight = repurchase_rank_weight
        if itemcf_rank_weight is not None:
            self.itemcf_rank_weight = itemcf_rank_weight
        if popularity_rank_weight is not None:
            self.popularity_rank_weight = popularity_rank_weight

        # 重排
        final_recs = {}
        for user_id in tqdm(self.val_users, desc="换权重重排", unit="用户"):
            final_recs[user_id] = self._rank_candidates_by_fusion(all_user_source_info[user_id])

        # 恢复旧权重
        for k, v in old.items():
            setattr(self, k, v)

        return final_recs

    def grid_search_weights(
        self,
        val_truth_dict: Dict[str, List],
        weight_grid: List[Dict],
        k: int = 12,
    ) -> List[Dict]:
        """网格搜索融合权重。

        对同一批候选集（跑一次召回），尝试不同权重组合，返回各组合的 MAP@k。

        weight_grid: [{"repurchase_weight": 3.0, "itemcf_weight": 1.5, ...}, ...]
        只传要改的权重即可，未传的保持当前值。
        """
        from src.metrics import calculate_map_at_k

        # Step 1: 跑一次召回
        print("\n📡 跑一次召回获取候选集...", flush=True)
        popularity_recs = self.popularity_recaller.recall_with_ranks(self.val_users)
        repurchase_recs = self.repurchase_recaller.recall_with_ranks(self.val_users)
        itemcf_recs = self.itemcf_recaller.recall_with_ranks(self.val_users)

        # Step 2: 构建 source_info（一次，最重的一步）
        print("\n📦 构建来源信息（一次，供后续复用）...", flush=True)
        all_user_source_info = self._build_all_user_source_info(
            popularity_recs, repurchase_recs, itemcf_recs
        )

        # Step 3: 对每个权重组合重新排序 + 评估
        results = []
        for i, weights in enumerate(weight_grid):
            print(f"\n[{i+1}/{len(weight_grid)}] 权重: {weights}", flush=True)
            final_recs = self._rerank_with_weights(all_user_source_info, **weights)
            map_score = calculate_map_at_k(val_truth_dict, final_recs, k=k)
            print(f"   MAP@{k} = {map_score:.6f}", flush=True)
            results.append({**weights, "map_at_k": map_score})

        # 恢复原始权重
        print("\n✅ 网格搜索完成", flush=True)
        return results


class PopularityRecall:
    """热门商品召回，支持多周指数衰减加权。"""

    def __init__(self, train_trans: pd.DataFrame, top_n: int = 50, decay: float = 0.7):
        self.train_trans = train_trans.copy()
        self.top_n = top_n
        self.decay = decay
        self.popular_items = self._get_popular_items_weighted()

    def _get_popular_items_weighted(self) -> List:
        """多周指数衰减加权：score(item) = Σ exp(-decay × week_diff) × count_in_week"""
        print("计算热门商品（多周指数加权）...", flush=True)
        max_week = self.train_trans["week"].max()
        sales = (
            self.train_trans.groupby(["article_id", "week"]).size()
            .reset_index(name="cnt")
        )
        sales["weight"] = np.exp(-self.decay * (max_week - sales["week"]))
        sales["score"] = sales["cnt"] * sales["weight"]
        item_scores = sales.groupby("article_id")["score"].sum().sort_values(ascending=False)
        return item_scores.head(self.top_n).index.tolist()

    def recall(self, val_users: List[str]) -> Dict[str, List]:
        return {user_id: self.popular_items[:] for user_id in val_users}

    def recall_with_ranks(self, val_users: List[str]) -> Dict[str, List]:
        return {user_id: self.popular_items[:] for user_id in val_users}


class RepurchaseRecall:
    """复购召回：返回用户历史购买记录（去重、倒序）。"""

    def __init__(self, train_trans: pd.DataFrame):
        self.train_trans = train_trans.copy()
        self.train_trans["customer_id"] = self.train_trans["customer_id"].astype(str)
        self.user_purchases = self._get_user_purchases()

    def _get_user_purchases(self) -> Dict[str, List]:
        """用 dict.fromkeys 保持顺序且去重，纯 C 代码，比 for 循环快 5-10 倍。"""
        print("计算用户购买记录（按时间排序）...", flush=True)
        sorted_trans = self.train_trans.sort_values("t_dat", ascending=False)
        return (
            sorted_trans.groupby("customer_id")["article_id"]
            .agg(lambda x: list(dict.fromkeys(x)))
            .to_dict()
        )

    def recall(self, val_users: List[str]) -> Dict[str, List]:
        return {
            user_id: self.user_purchases.get(user_id, [])
            for user_id in tqdm(val_users, desc="复购召回", unit="用户")
        }

    def recall_with_ranks(self, val_users: List[str]) -> Dict[str, List]:
        return {
            user_id: self.user_purchases.get(user_id, [])
            for user_id in tqdm(val_users, desc="复购召回（带排名）", unit="用户")
        }


class ItemCFRecallSparse:
    """基于稀疏矩阵的 ItemCF。

    - 相似度矩阵：二值行为算 Jaccard
    - 用户打分向量：时间衰减，近行为权重更高
    """

    def __init__(self, train_trans: pd.DataFrame, top_k: int = 20):
        self.train_trans = train_trans.copy()
        self.train_trans["customer_id"] = self.train_trans["customer_id"].astype(str)
        self.top_k = top_k
        self.user2idx = {}
        self.item2idx = {}
        self.idx2item = {}
        self.user_item_binary = None
        self.user_item_weighted = None
        self.similarity_matrix = None
        self._prepare_matrices()

    def _prepare_matrices(self):
        print_memory_usage("ItemCF矩阵构建开始")

        self.user_ids = self.train_trans["customer_id"].unique()
        self.item_ids = self.train_trans["article_id"].unique()

        self.user2idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item2idx = {item: idx for idx, item in enumerate(self.item_ids)}
        self.idx2item = {idx: item for idx, item in enumerate(self.item_ids)}

        rows = self.train_trans["customer_id"].map(self.user2idx).values
        cols = self.train_trans["article_id"].map(self.item2idx).values

        binary_data = np.ones(len(self.train_trans), dtype=np.float32)
        weighted_data = (1.0 / (1.0 + self.train_trans["week"].values)).astype(np.float32)

        self.user_item_binary = csr_matrix(
            (binary_data, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
        )
        self.user_item_binary.sum_duplicates()
        self.user_item_binary.data[:] = 1.0

        self.user_item_weighted = csr_matrix(
            (weighted_data, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
        )
        self.user_item_weighted.sum_duplicates()

        print("计算商品共现矩阵...", flush=True)
        co_occur_matrix = self.user_item_binary.T.dot(self.user_item_binary).tocsr()
        co_occur_matrix.setdiag(0)
        co_occur_matrix.eliminate_zeros()

        print("计算余弦相似度（热门商品降权）...", flush=True)
        coo = co_occur_matrix.tocoo()
        item_user_counts = np.asarray(self.user_item_binary.sum(axis=0)).ravel()
        # 余弦：co_occur(i,j) / sqrt(count(i) * count(j))
        denom = np.sqrt(item_user_counts[coo.row] * item_user_counts[coo.col]).astype(np.float32)
        sim_data = np.divide(
            coo.data.astype(np.float32),
            denom,
            out=np.zeros_like(coo.data, dtype=np.float32),
            where=denom > 1e-8
        )

        self.similarity_matrix = csr_matrix(
            (sim_data, (coo.row, coo.col)),
            shape=co_occur_matrix.shape,
        )
        print_memory_usage("ItemCF矩阵构建完成")

    def _recall_internal(self, val_users: List[str]) -> Dict[str, List]:
        print("执行ItemCF召回...", flush=True)
        itemcf_recs = {}
        batch_size = 1000

        for i in tqdm(range(0, len(val_users), batch_size), desc="批量ItemCF召回", unit="批"):
            batch_users = val_users[i:i + batch_size]

            for user_id in batch_users:
                if user_id not in self.user2idx:
                    itemcf_recs[user_id] = []
                    continue

                u_idx = self.user2idx[user_id]
                user_vector_weighted = self.user_item_weighted[u_idx]
                user_vector_binary = self.user_item_binary[u_idx]

                scores = user_vector_weighted.dot(self.similarity_matrix).toarray().ravel()
                purchased_indices = user_vector_binary.indices
                scores[purchased_indices] = 0.0

                non_zero_count = np.count_nonzero(scores)
                if non_zero_count == 0:
                    itemcf_recs[user_id] = []
                    continue

                k = min(self.top_k, non_zero_count)
                top_k_indices = np.argpartition(-scores, k - 1)[:k]
                top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]
                itemcf_recs[user_id] = [self.idx2item[int(idx)] for idx in top_k_indices]

        return itemcf_recs

    def recall(self, val_users: List[str]) -> Dict[str, List]:
        return self._recall_internal(val_users)

    def recall_with_ranks(self, val_users: List[str]) -> Dict[str, List]:
        return self._recall_internal(val_users)


class ItemCFRecallDict:
    """兼容旧接口，直接复用稳定的 sparse 版本。"""

    def __init__(self, train_trans: pd.DataFrame, top_k: int = 20):
        self.delegate = ItemCFRecallSparse(train_trans, top_k)

    def recall(self, val_users: List[str]) -> Dict[str, List]:
        return self.delegate.recall(val_users)

    def recall_with_ranks(self, val_users: List[str]) -> Dict[str, List]:
        return self.delegate.recall_with_ranks(val_users)