import csv
import random
import os
from datetime import datetime, timedelta

def get_last_week_active_users():
    """
    第一遍遍历：寻找 1000 个目标用户
    """
    print("第一遍遍历：寻找目标用户...", flush=True)
    
    # 计算最后一周的日期（使用固定日期，避免依赖当前时间）
    last_week_start = '2020-09-16'  # 最后一周开始日期
    target_users = set()
    
    print(f"寻找 {last_week_start} 之后有购买行为的用户", flush=True)
    
    with open('data/transactions_train.csv', 'r') as f:
        # 读取表头
        header = f.readline().strip()
        print(f"表头: {header}", flush=True)
        
        # 逐行读取
        for line in f:
            # 解析行数据
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
                
            t_dat = parts[0]  # 日期
            customer_id = parts[1]  # 用户ID
            
            # 如果日期属于最后一周，则将该用户加入目标用户集合
            if t_dat >= last_week_start:
                target_users.add(customer_id)
                
                # 关键拦截：只要达到1000个用户就立即终止
                if len(target_users) == 1000:
                    print(f"已找到 {len(target_users)} 个目标用户，终止第一遍遍历", flush=True)
                    break
    
    print(f"第一遍遍历完成，共找到 {len(target_users)} 个目标用户", flush=True)
    return target_users

def extract_user_transactions(target_users):
    """
    第二遍遍历：提取这 1000 人的所有历史记录
    """
    print("第二遍遍历：提取目标用户的历史记录...", flush=True)
    
    output_file = 'sampled_1000_transactions.csv'
    
    # 检查输出文件是否已存在，如果存在则删除
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"删除已存在的输出文件: {output_file}", flush=True)
    
    with open('data/transactions_train.csv', 'r') as fin:
        with open(output_file, 'w') as fout:
            # 读取表头并写入新文件
            header = fin.readline().strip()
            fout.write(header + '\n')
            print(f"表头已写入: {header}", flush=True)
            
            # 逐行读取原文件
            line_count = 0
            matched_count = 0
            
            for line in fin:
                line_count += 1
                
                # 解析行数据
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                    
                customer_id = parts[1]  # 用户ID
                
                # 如果这一行的用户ID在目标用户集合中，就写入新文件
                if customer_id in target_users:
                    fout.write(line)
                    matched_count += 1
                    
                    # 每1000行打印一次进度
                    if matched_count % 1000 == 0:
                        print(f"已处理 {line_count} 行，匹配 {matched_count} 条记录", flush=True)
    
    print(f"第二遍遍历完成，共处理 {line_count} 行，匹配 {matched_count} 条记录", flush=True)
    print(f"目标用户的历史记录已保存到: {output_file}", flush=True)

def main():
    print("开始创建中等规模样本（1000个用户）...", flush=True)
    print("使用纯原生 Python 流式处理，避免内存溢出", flush=True)
    
    # 确保数据文件存在
    if not os.path.exists('data/transactions_train.csv'):
        print("错误：transactions_train.csv 文件不存在！", flush=True)
        return
    
    # 第一遍遍历：寻找目标用户
    target_users = get_last_week_active_users()
    
    if len(target_users) == 0:
        print("警告：没有找到任何目标用户！", flush=True)
        return
    
    # 第二遍遍历：提取历史记录
    extract_user_transactions(target_users)
    
    print("中等规模样本创建完成！", flush=True)

if __name__ == "__main__":
    main()