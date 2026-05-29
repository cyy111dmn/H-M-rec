import warnings

import numpy as np
import pandas as pd
from typing import Dict, List, Union

from src.utils import normalize_listlike


def _normalize_items(x):
    """统一转成 list（委托 utils.normalize_listlike，保留此函数兼容旧调用）。"""
    return normalize_listlike(x)


def calculate_ap_at_k(actual: List[int], predicted: List[int], k: int = 12) -> float:
    """
    计算单个用户的 Average Precision at K。
    对于没有真实购买的用户，返回 0.0。
    """
    actual = _normalize_items(actual)
    predicted = _normalize_items(predicted)

    if len(actual) == 0:
        return 0.0

    predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    actual_set = set(actual)

    for i, p in enumerate(predicted):
        if p in actual_set and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def calculate_map_at_k(
    actual_data: Union[pd.DataFrame, Dict[int, List[int]]],
    predicted_dict: Dict[int, List[int]],
    k: int = 12,
    exclude_empty_users: bool = True
) -> float:
    """
    计算所有用户的 Mean Average Precision at K。

    Args:
        actual_data: 真实购买记录（DataFrame 或 dict）
        predicted_dict: 预测结果
        k: 截断位置
        exclude_empty_users: 是否排除验证期内无购买的用户。
                             Kaggle 官方评估排除这些用户，设为 True 以对齐线上分数。

    Key points:
    1. exclude_empty_users=True（默认）时，无购买用户不参与平均，与 Kaggle 官方一致
    2. 没有预测结果的用户按空预测处理
    3. 兼容 parquet 读出来的 ndarray/list 混合格式
    """
    if isinstance(actual_data, pd.DataFrame):
        if actual_data is None or actual_data.empty:
            return 0.0
        actual_dict = dict(zip(actual_data['customer_id'], actual_data['purchased_articles']))
    else:
        actual_dict = actual_data or {}

    if not actual_dict:
        return 0.0

    ap_scores = []
    for user_id, actual_items in actual_dict.items():
        actual_items = _normalize_items(actual_items)

        # Kaggle: 无购买用户不参与分数计算
        if exclude_empty_users and len(actual_items) == 0:
            continue

        predicted_items = _normalize_items(predicted_dict.get(user_id, []))
        ap = calculate_ap_at_k(actual_items, predicted_items, k)
        ap_scores.append(ap)

    return float(np.mean(ap_scores)) if ap_scores else 0.0


def evaluate_predictions(
    submission_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    k: int = 12
) -> Dict[str, float]:
    """
    评估预测结果。
    """
    submission_df = submission_df.copy()
    submission_df['prediction'] = submission_df['prediction'].apply(
        lambda x: [int(item) for item in str(x).split()]
    )
    predicted_dict = dict(zip(submission_df['customer_id'], submission_df['prediction']))

    if 'purchased_articles' in ground_truth_df.columns:
        actual_dict = dict(zip(ground_truth_df['customer_id'], ground_truth_df['purchased_articles']))
    else:
        actual_dict = ground_truth_df.groupby('customer_id')['article_id'].apply(list).to_dict()

    map_at_k = calculate_map_at_k(actual_dict, predicted_dict, k)

    return {
        'map_at_k': map_at_k,
        'k': k,
        'num_users': len(actual_dict),
        'num_users_with_predictions': len(predicted_dict)
    }


def print_evaluation_results(results: Dict[str, float]):
    print(f"\n📊 === 评估结果 ===")
    print(f"MAP@{results['k']}: {results['map_at_k']:.6f}")
    print(f"评估用户数: {results['num_users']}")
    print(f"有预测的用户数: {results['num_users_with_predictions']}")
    print("==================\n")