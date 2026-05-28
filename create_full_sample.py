import pandas as pd
import numpy as np
import os
import psutil
import tqdm
from tqdm import tqdm

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024 / 1024  # GB

def print_memory_usage(step):
    """打印内存使用情况"""
    memory_gb = get_memory_usage()
    print(f"[内存监控] {step}: {memory_gb:.2f} GB", flush=True)

def main():
    print("=" * 60)
    print("开始全量高质量采样 - 10万活跃用户")
    print("=" * 60)
    print("正在读取数据...", flush=True)
    print_memory_usage("开始")
    
    # 数据目录
    data_dir = 'data'
    
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)
    
    # 先查看数据结构（只读取前5行）
    print("查看数据结构...", flush=True)
    sample_data = pd.read_csv(f'{data_dir}/transactions_train.csv', nrows=5)
    print(f"数据列名: {list(sample_data.columns)}")
    print(f"数据形状: {sample_data.shape}")
    print_memory_usage("数据结构查看完成")
    
    # 使用分块读取的方式处理大文件 - 增大块大小以利用充足内存
    print("开始分块读取交易数据...", flush=True)
    chunk_size = 1000000  # 100万行一块，充分利用60GB内存
    chunks = pd.read_csv(f'{data_dir}/transactions_train.csv', chunksize=chunk_size)
    transactions_list = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="读取数据块", unit="块", total=None)):
        transactions_list.append(chunk)
        if (i + 1) % 5 == 0:  # 每5个块打印一次进度
            print(f"已加载 {len(transactions_list)} 个数据块，总行数: {sum(len(df) for df in transactions_list):,}", flush=True)
            print_memory_usage(f"加载 {len(transactions_list)} 块后")
    
    # 合并所有数据块
    print("合并数据块...", flush=True)
    transactions = pd.concat(transactions_list, ignore_index=True)
    print(f"合并后的交易数据形状: {transactions.shape}", flush=True)
    print_memory_usage("数据合并完成")
    
    # 加载客户数据
    print("加载客户数据...", flush=True)
    customers = pd.read_csv(f'{data_dir}/customers.csv')
    print(f"客户数据形状: {customers.shape}")
    print_memory_usage("客户数据加载完成")
    
    # 加载商品数据
    print("加载商品数据...", flush=True)
    articles = pd.read_csv(f'{data_dir}/articles.csv')
    print(f"商品数据形状: {articles.shape}")
    print_memory_usage("商品数据加载完成")
    
    # 统计每个用户的购买次数
    print("统计用户购买次数...", flush=True)
    user_trans_counts = transactions.groupby('customer_id').size()
    print(f"总用户数: {len(user_trans_counts):,}")
    print(f"总交易数: {len(transactions):,}")
    print(f"平均每个用户购买次数: {user_trans_counts.mean():.2f}")
    print(f"购买次数中位数: {user_trans_counts.median():.2f}")
    print_memory_usage("用户购买次数统计完成")
    
    # 筱选出购买次数最多的前100,000名活跃用户
    print("筛选前100,000名活跃用户...", flush=True)
    # 按购买次数降序排序
    sorted_users = user_trans_counts.sort_values(ascending=False)
    top_100k_users = sorted_users.head(100000).index.tolist()
    print(f"前100,000名活跃用户的购买次数范围: {sorted_users.head(1).values[0]} 到 {sorted_users.iloc[99999]:.0f}")
    print(f"前100,000名用户总购买次数: {sorted_users.head(100000).sum():,}")
    print_memory_usage("活跃用户筛选完成")
    
    # 提取这100,000名用户的所有交易记录
    print("提取活跃用户交易记录...", flush=True)
    sampled_trans = transactions[transactions['customer_id'].isin(top_100k_users)].copy()
    
    # 提取对应的客户信息
    sampled_customers = customers[customers['customer_id'].isin(top_100k_users)].copy()
    
    # 提取对应的商品信息
    sampled_articles = articles[articles['article_id'].isin(sampled_trans['article_id'].unique())].copy()
    
    # 保存采样数据
    print("保存采样数据...", flush=True)
    
    # 保存交易数据
    sampled_trans.to_csv(f'{data_dir}/transactions_sample_10w.csv', index=False)
    print(f"交易数据已保存: {data_dir}/transactions_sample_10w.csv ({len(sampled_trans):,} 行)")
    
    # 保存客户数据
    sampled_customers.to_csv(f'{data_dir}/customers_sample_10w.csv', index=False)
    print(f"客户数据已保存: {data_dir}/customers_sample_10w.csv ({len(sampled_customers):,} 行)")
    
    # 保存商品数据
    sampled_articles.to_csv(f'{data_dir}/articles_sample_10w.csv', index=False)
    print(f"商品数据已保存: {data_dir}/articles_sample_10w.csv ({len(sampled_articles):,} 行)")
    
    print("\n" + "=" * 60)
    print("全量高质量采样完成！")
    print("=" * 60)
    print(f"采样数据统计:")
    print(f"- 活跃用户数: {len(top_100k_users):,}")
    print(f"- 交易记录数: {len(sampled_trans):,}")
    print(f"- 客户数: {len(sampled_customers):,}")
    print(f"- 商品数: {len(sampled_articles):,}")
    print(f"- 数据已保存到: {data_dir}/")
    print_memory_usage("采样完成")

if __name__ == "__main__":
    main()