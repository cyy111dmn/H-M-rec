import pandas as pd
import numpy as np
import gc
import os
from typing import Tuple, Dict

def load_parquet_data(data_dir):
    print("📂 加载全量数据...", flush=True)
    
    # 优先尝试加载 CSV 格式的全量数据，如果不存在则加载 Parquet 格式
    try:
        transactions = pd.read_csv(f'{data_dir}/transactions_train.csv')
        customers = pd.read_csv(f'{data_dir}/customers.csv')[['customer_id', 'age']]
        print("✅ 使用 CSV 格式全量数据")
    except FileNotFoundError:
        print("⚠️ CSV 文件不存在，尝试加载 Parquet 格式...")
        transactions = pd.read_parquet(f'{data_dir}/transactions_train.parquet')
        customers = pd.read_parquet(f'{data_dir}/customers.parquet')[['customer_id', 'age']]
        print("✅ 使用 Parquet 格式数据")
    
    print(f"📊 实际读取到: {len(customers)} 用户, {len(transactions)} 交易记录")

    # ==========================================
    # 【核心修复】防数据污染的安全拦截墙
    # ==========================================
    if len(transactions) < 10000000:
        raise ValueError(f"❌ 交易数据量异常！只加载了 {len(transactions)} 条（应为3100万+），疑似读取了采样数据！")
        
    if len(customers) < 1000000:
        # 如果用户数不对，自动扫描文件夹帮用户找回全量文件
        try:
            files = os.listdir(data_dir)
            cust_files = [f for f in files if 'customer' in f.lower()]
            sizes = {f: os.path.getsize(os.path.join(data_dir, f)) / (1024*1024) for f in cust_files}
            
            err_msg = f"\n❌ 致命错误：用户数据量严重缺失！当前只加载了 {len(customers)} 个用户，这肯定是之前的采样文件！\n"
            err_msg += f"🔍 正在帮你扫描 {data_dir} 目录下的嫌疑文件大小(MB)：\n"
            for f, s in sizes.items():
                err_msg += f"   📄 {f}: {s:.2f} MB\n"
            err_msg += "\n💡 提示：全量用户文件 (137万行) 的体积应该在 50MB - 200MB 之间（取决于格式）。\n"
            err_msg += "👉 请在终端手动把你真实的全量文件重命名为 `customers.csv`，或者修改这里的读取路径！"
            raise ValueError(err_msg)
        except Exception as e:
            raise ValueError(f"❌ 用户数据量异常（仅 {len(customers)} 行），且扫描目录失败: {e}")

    # Radek 官方处理：填充年龄缺失值
    median_age = customers['age'].median()
    customers['age'] = customers['age'].fillna(median_age).astype('int8')
    
    return transactions, customers

def load_id_mapping(data_dir):
    print("🔗 加载 ID 映射字典...", flush=True)
    raw_cust = pd.read_csv(f'{data_dir}/customers.csv', usecols=['customer_id'])
    
    # 如果映射字典的底表也被污染了，同样拦截
    if len(raw_cust) < 1000000:
        raise ValueError(f"❌ ID映射底表加载失败：文件只有 {len(raw_cust)} 行，请确保 {data_dir}/customers.csv 是全量文件！")
        
    # 【核心修复】不依赖 Pandas 的 dtype，直接拿第一条数据做绝对判断！
    first_val = raw_cust['customer_id'].iloc[0]
    
    if isinstance(first_val, str):
        # 如果是字符串（16进制），截取最后16位转成整型，大幅节省内存
        raw_cust['uint_id'] = raw_cust['customer_id'].apply(lambda x: int(x[-16:], 16)).astype('uint64')
    else:
        # 如果已经是数字，直接转
        raw_cust['uint_id'] = raw_cust['customer_id'].astype('uint64')
        
    uint_to_hex_cust = dict(zip(raw_cust['uint_id'], raw_cust['customer_id']))
    del raw_cust
    gc.collect()
    return uint_to_hex_cust

def split_data_by_time(transactions: pd.DataFrame,
                      customers: pd.DataFrame,
                      validation_weeks: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间切分数据，使用最后几周作为验证集
    """
    print("🔪 按时间切分数据...", flush=True)
    
    # 计算时间窗口
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    max_date = transactions['t_dat'].max()
    transactions['week'] = (max_date - transactions['t_dat']).dt.days // 7
    
    # 获取最大周数
    max_week = transactions['week'].min()
    
    # 划分训练集和验证集
    train_trans = transactions[transactions['week'] > validation_weeks].copy()
    val_trans = transactions[transactions['week'] <= validation_weeks].copy()
    
    # 获取验证集中的用户
    val_users = val_trans['customer_id'].unique()
    val_customers = customers[customers['customer_id'].isin(val_users)].copy()
    
    print(f"  - 训练集时间范围: 周 {train_trans['week'].min()} 到 周 {train_trans['week'].max()}")
    print(f"  - 验证集时间范围: 周 {val_trans['week'].min()} 到 周 {val_trans['week'].max()}")
    print(f"  - 训练集交易数: {len(train_trans)}")
    print(f"  - 验证集交易数: {len(val_trans)}")
    print(f"  - 验证集用户数: {len(val_customers)}")
    
    return train_trans, val_trans, val_customers

def prepare_validation_data(val_trans: pd.DataFrame) -> pd.DataFrame:
    """
    准备验证数据，生成每个用户在验证期间购买的商品列表
    """
    print("📝 准备验证数据 Ground Truth...", flush=True)
    
    # 获取每个用户在验证期间购买的商品
    val_ground_truth = val_trans.groupby('customer_id')['article_id'].apply(list).reset_index()
    val_ground_truth.columns = ['customer_id', 'purchased_articles']
    
    return val_ground_truth