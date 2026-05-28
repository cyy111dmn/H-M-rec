import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict
import gc

def sample_users(transactions: pd.DataFrame, 
                customers: pd.DataFrame, 
                sample_ratio: float = 0.05,
                random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按比例采样用户，并返回对应的交易和客户数据
    
    Args:
        transactions: 交易数据
        customers: 客户数据
        sample_ratio: 采样比例，默认为0.05（5%）
        random_seed: 随机种子，确保可重复性
        
    Returns:
        sampled_trans: 采样后的交易数据
        sampled_customers: 采样后的客户数据
    """
    print(f"采样 {sample_ratio*100:.1f}% 的用户数据...", flush=True)
    
    # 获取所有唯一用户
    all_users = customers['customer_id'].unique()
    np.random.seed(random_seed)
    
    # 随机采样用户
    sample_size = int(len(all_users) * sample_ratio)
    sampled_users = np.random.choice(all_users, size=sample_size, replace=False)
    
    # 筛选采样用户的交易记录
    sampled_trans = transactions[transactions['customer_id'].isin(sampled_users)].copy()
    
    # 筛选采样用户的信息
    sampled_customers = customers[customers['customer_id'].isin(sampled_users)].copy()
    
    print(f"原始用户数: {len(all_users)}")
    print(f"采样用户数: {len(sampled_users)}")
    print(f"原始交易数: {len(transactions)}")
    print(f"采样交易数: {len(sampled_trans)}")
    
    return sampled_trans, sampled_customers

def create_sampled_dataset(data_dir: str, 
                         output_dir: str,
                         sample_ratio: float = 0.05,
                         random_seed: int = 42):
    """
    创建采样数据集并保存到文件
    
    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        sample_ratio: 采样比例，默认为0.05（5%）
        random_seed: 随机种子，确保可重复性
    """
    print("创建采样数据集...", flush=True)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始数据
    transactions = pd.read_parquet(f'{data_dir}/transactions_train.parquet')
    customers = pd.read_parquet(f'{data_dir}/customers.parquet')[['customer_id', 'age']]
    
    # 采样用户
    sampled_trans, sampled_customers = sample_users(transactions, customers, sample_ratio, random_seed)
    
    # 保存采样数据
    sampled_trans.to_parquet(f'{output_dir}/transactions_sample.parquet')
    sampled_customers.to_parquet(f'{output_dir}/customers_sample.parquet')
    
    print(f"采样数据已保存到: {output_dir}")
    print(f"文件大小: transactions_sample.parquet ({sampled_trans.memory_usage(deep=True).sum() / 1024**2:.2f} MB)")
    print(f"文件大小: customers_sample.parquet ({sampled_customers.memory_usage(deep=True).sum() / 1024**2:.2f} MB)")
    
    # 清理内存
    del transactions, customers, sampled_trans, sampled_customers
    gc.collect()

def load_sampled_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载采样数据
    
    Args:
        data_dir: 数据目录
        
    Returns:
        transactions: 交易数据
        customers: 客户数据
    """
    print("加载采样数据...", flush=True)
    
    transactions = pd.read_parquet(f'{data_dir}/transactions_sample.parquet')
    customers = pd.read_parquet(f'{data_dir}/customers_sample.parquet')
    
    # Radek 官方处理：填充年龄缺失值
    median_age = customers['age'].median()
    customers['age'] = customers['age'].fillna(median_age).astype('int8')
    
    return transactions, customers

def validate_sampled_data(transactions: pd.DataFrame, 
                         customers: pd.DataFrame,
                         min_transactions_per_user: int = 1) -> Dict:
    """
    验证采样数据的质量
    
    Args:
        transactions: 交易数据
        customers: 客户数据
        min_transactions_per_user: 每个用户的最小交易数
        
    Returns:
        包含验证统计信息的字典
    """
    print("验证采样数据质量...", flush=True)
    
    # 统计信息
    stats = {
        'num_customers': len(customers),
        'num_transactions': len(transactions),
        'num_articles': transactions['article_id'].nunique(),
        'date_range': {
            'start': transactions['t_dat'].min(),
            'end': transactions['t_dat'].max()
        }
    }
    
    # 每个用户的交易数统计
    user_trans_counts = transactions.groupby('customer_id').size()
    stats['user_transaction_stats'] = {
        'mean': user_trans_counts.mean(),
        'median': user_trans_counts.median(),
        'min': user_trans_counts.min(),
        'max': user_trans_counts.max(),
        'users_with_min_trans': (user_trans_counts >= min_transactions_per_user).sum()
    }
    
    # 年龄分布
    stats['age_stats'] = {
        'mean': customers['age'].mean(),
        'median': customers['age'].median(),
        'min': customers['age'].min(),
        'max': customers['age'].max()
    }
    
    # 打印统计信息
    print(f"\n📊 === 采样数据统计 ===")
    print(f"客户数: {stats['num_customers']}")
    print(f"交易数: {stats['num_transactions']}")
    print(f"商品数: {stats['num_articles']}")
    print(f"日期范围: {stats['date_range']['start']} 到 {stats['date_range']['end']}")
    print(f"用户交易数 - 均值: {stats['user_transaction_stats']['mean']:.2f}, 中位数: {stats['user_transaction_stats']['median']:.0f}")
    print(f"用户交易数 - 最小: {stats['user_transaction_stats']['min']}, 最大: {stats['user_transaction_stats']['max']}")
    print(f"满足最小交易数({min_transactions_per_user})的用户数: {stats['user_transaction_stats']['users_with_min_trans']}")
    print(f"年龄 - 均值: {stats['age_stats']['mean']:.1f}, 中位数: {stats['age_stats']['median']:.0f}")
    print(f"年龄 - 最小: {stats['age_stats']['min']}, 最大: {stats['age_stats']['max']}")
    print("====================\n")
    
    return stats