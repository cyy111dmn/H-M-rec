import numpy as np
import pandas as pd
from typing import Dict, List, Union


def _normalize_items(x):
    """
    把真实标签/预测标签统一转成 Python list，兼容：
    - list
    - tuple
    - set
    - numpy.ndarray
    - pandas Series
    - NaN / None
    """
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, set):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, pd.Series):
        return x.tolist()
    return [x]


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
    k: int = 12
) -> float:
    """
    计算所有用户的 Mean Average Precision at K。

    关键点：
    1. 空标签用户按 0 分计入平均值
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