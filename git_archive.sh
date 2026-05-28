#!/bin/bash

# Git 存档脚本
# 用于保存 H&M 推荐系统的工作

echo "🚀 开始 Git 存档..."

# 检查是否在 git 仓库中
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "❌ 错误: 当前目录不是 Git 仓库"
    echo "正在初始化 Git 仓库..."
    git init
fi

# 添加所有文件到暂存区
echo "📁 添加文件到暂存区..."
git add .

# 检查是否有更改
if git diff --cached --quiet; then
    echo "⚠️  没有需要提交的更改"
    exit 0
fi

# 提交更改
echo "💾 提交更改..."
git commit -m "feat: 实现 H&M 推荐系统 - 10,000用户规模优化版本

- 优化采样规模从20,000人调整为10,000人
- 改进ItemCF内存使用，避免全量密集矩阵
- 添加内存监控和进度条显示
- 实现多路召回对比实验
- 添加纯热门Baseline对比
- 优化CPU环境下的性能表现

性能优化:
- 使用分块读取处理大文件
- 采用双层dict存储相似度矩阵
- 添加tqdm进度条
- 实时内存监控
- 向量化操作优化"

# 创建标签
echo "🏷️  创建版本标签..."
git tag -a "v1.0-medium" -m "10,000用户规模优化版本"

# 显示提交信息
echo "✅ Git 存档完成！"
echo "📊 提交信息:"
git log --oneline -1

echo "🏷️  标签信息:"
git tag -l "v1.0-medium"

echo "📁 当前状态:"
git status

echo "🎉 所有工作已成功保存到 Git！"