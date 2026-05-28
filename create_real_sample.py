import pandas as pd
import numpy as np
import os
from src.sampling import sample_users, validate_sampled_data

def main():
    print("从真实 H&M 全量数据中创建 20,000 名活跃用户的采样...", flush=True)
    
    # 数据目录
    data_dir = 'data'
    sample_dir = f'{data_dir}/sample_20k'
    
    # 确保采样目录存在
    os.makedirs(sample_dir, exist_ok=True)
    
    # 加载原始真实数据
    print("加载原始真实数据...", flush=True)
    # 使用分块读取的方式处理大文件
    chunks = pd.read_csv(f'{data_dir}/transactions_train.csv', chunksize=1000000)
    transactions_list = []
    
    for chunk in chunks:
        transactions_list.append(chunk)
        print(f"已加载 {len(transactions_list)} 个数据块，最新块大小: {chunk.shape}", flush=True)
    
    # 合并所有数据块
    transactions = pd.concat(transactions_list, ignore_index=True)
    print(f"合并后的交易数据形状: {transactions.shape}", flush=True)
    
    # 加载客户数据
    customers = pd.read_csv(f'{data_dir}/customers.csv')
    
    # 填充年龄缺失值
    median_age = customers['age'].median()
    customers['age'] = customers['age'].fillna(median_age).astype('int8')
    
    # 计算活跃用户（交易次数 >= 5）
    user_trans_counts = transactions.groupby('customer_id').size()
    active_users = user_trans_counts[user_trans_counts >= 5].index.tolist()
    print(f"活跃用户数（交易次数 >= 5）: {len(active_users)}", flush=True)
    
    # 如果活跃用户数超过20,000，随机采样20,000个
    if len(active_users) > 20000:
        np.random.seed(42)
        sampled_active_users = np.random.choice(active_users, size=20000, replace=False)
        print(f"随机采样 20,000 个活跃用户", flush=True)
    else:
        sampled_active_users = active_users
        print(f"活跃用户数不足20,000，使用所有 {len(active_users)} 个活跃用户", flush=True)
    
    # 筛选采样用户的交易记录
    sampled_trans = transactions[transactions['customer_id'].isin(sampled_active_users)].copy()
    sampled_customers = customers[customers['customer_id'].isin(sampled_active_users)].copy()
    
    print(f"采样后交易数据形状: {sampled_trans.shape}", flush=True)
    print(f"采样后客户数据形状: {sampled_customers.shape}", flush=True)
    
    # 保存采样数据
    print("保存采样数据...", flush=True)
    sampled_trans.to_parquet(f'{sample_dir}/transactions_sample.parquet')
    sampled_customers.to_parquet(f'{sample_dir}/customers_sample.parquet')
    
    # 验证采样数据质量
    print("验证采样数据质量...", flush=True)
    validate_sampled_data(sampled_trans, sampled_customers)
    
    print(f"20,000 活跃用户采样数据已保存到: {sample_dir}")
    
    # 更新 config.py 以使用真实采样数据
    print("更新配置文件...", flush=True)
    update_config(sample_dir)
    
    print("✅ 真实数据采样创建完成！")

def update_config(sample_dir):
    """更新配置文件以使用真实采样数据"""
    config_path = 'src/config.py'
    
    # 读取当前配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    # 更新 DATA_DIR 为采样数据目录
    new_data_dir = f"DATA_DIR = '{sample_dir}'"
    config_content = config_content.replace(
        f"DATA_DIR = '{config_content.split('DATA_DIR = ')[1].split('\\n')[0][1:-1]}'",
        new_data_dir
    )
    
    # 确保 DEBUG_MODE 为 True
    config_content = config_content.replace(
        "DEBUG_MODE = False",
        "DEBUG_MODE = True"
    )
    
    # 写回配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"配置文件已更新: DATA_DIR = {sample_dir}")

if __name__ == "__main__":
    main()