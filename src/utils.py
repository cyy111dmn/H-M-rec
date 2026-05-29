#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工具函数合集：内存监控、数据类型压缩、列表归一化。
"""

from typing import Any, List

import numpy as np
import pandas as pd
import psutil


def get_memory_usage() -> float:
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def print_memory_usage(step: str):
    print(f"[内存监控] {step}: {get_memory_usage():.2f} MB", flush=True)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type).startswith("int"):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"内存压缩: {start_mem:.2f} MB -> {end_mem:.2f} MB", flush=True)
    return df


def normalize_listlike(x: Any) -> List:
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, set):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, pd.Series):
        return x.tolist()
    import warnings
    warnings.warn(f"normalize_listlike: 未预期的类型 {type(x)}，当作单元素列表处理", stacklevel=2)
    return [x]
