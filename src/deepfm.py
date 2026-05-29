#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepFM 模型 — PyTorch 实现

架构：
- FM 部分: 一阶线性 + 二阶 pairwise inner product
- Deep 部分: MLP 学习高阶交互
- 输出: sigmoid(FM + Deep)

主要用于替代 LGBM LambdaRank 做精排。
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFM(nn.Module):
    """DeepFM 排序模型。

    Args:
        cat_feat_info: 类别特征信息，Dict[特征名, 类别数]
        num_feat_names: 数值特征名列表
        embed_dim: 类别特征 embedding 维度
        mlp_dims: Deep 部分 MLP 各层维度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        cat_feat_info: Dict[str, int],
        num_feat_names: List[str],
        embed_dim: int = 16,
        mlp_dims: List[int] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cat_feat_names = list(cat_feat_info.keys())
        self.num_feat_names = num_feat_names
        self.embed_dim = embed_dim
        self.num_cat = len(self.cat_feat_names)
        self.num_num = len(num_feat_names)

        # --- Embedding 层 ---
        # 每个类别特征对应一个 embedding 表
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embed_dim, padding_idx=0)
            for name, vocab_size in cat_feat_info.items()
        })

        # FM 一阶: 每个类别特征的线性偏置 (bias term)
        self.fm_first_order = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=1, padding_idx=0)
            for name, vocab_size in cat_feat_info.items()
        })

        # FM 一阶: 数值特征的线性权重
        self.fm_num_linear = nn.Linear(self.num_num, 1, bias=False)

        # FM 整体偏置
        self.fm_bias = nn.Parameter(torch.zeros(1))

        # --- Deep 部分 ---
        deep_input_dim = self.num_cat * embed_dim + self.num_num
        deep_layers = []
        in_dim = deep_input_dim
        for out_dim in mlp_dims:
            deep_layers.append(nn.Linear(in_dim, out_dim))
            deep_layers.append(nn.BatchNorm1d(out_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        deep_layers.append(nn.Linear(in_dim, 1))
        self.deep_mlp = nn.Sequential(*deep_layers)

        self._init_weights()

    def _init_weights(self):
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.fm_first_order.values():
            nn.init.xavier_uniform_(emb.weight)
        for m in self.deep_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_cat_embeddings(self, cat_x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取类别特征的 embedding 向量，拼成 [batch, num_cat, embed_dim]"""
        emb_list = []
        for name in self.cat_feat_names:
            emb = self.embeddings[name](cat_x[name])  # [batch, embed_dim]
            emb_list.append(emb)
        return torch.stack(emb_list, dim=1)  # [batch, num_cat, embed_dim]

    def forward(self, cat_x: Dict[str, torch.Tensor], num_x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            cat_x: 类别特征，Dict[特征名, Tensor[batch]]
            num_x: 数值特征，Tensor[batch, num_num]

        Returns:
            logits: Tensor[batch] — 未经 sigmoid 的原始分数
        """
        batch_size = num_x.size(0)

        # ========== FM 一阶 ==========
        first_order = self.fm_bias.expand(batch_size)  # 全局偏置
        # 类别特征一阶
        for name in self.cat_feat_names:
            first_order = first_order + self.fm_first_order[name](cat_x[name]).squeeze(-1)
        # 数值特征一阶
        first_order = first_order + self.fm_num_linear(num_x).squeeze(-1)

        # ========== FM 二阶 (pairwise inner product) ==========
        # 经典 FM 二阶高效计算: sum(f_i * f_j) = 0.5 * ( (sum f)^2 - sum(f^2) )
        cat_emb = self._get_cat_embeddings(cat_x)  # [batch, num_cat, embed_dim]
        square_of_sum = cat_emb.sum(dim=1).pow(2)  # [batch, embed_dim]
        sum_of_square = (cat_emb**2).sum(dim=1)    # [batch, embed_dim]
        second_order = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)  # [batch]

        # ========== Deep 部分 ==========
        cat_emb_flat = cat_emb.reshape(batch_size, -1)  # [batch, num_cat * embed_dim]
        deep_input = torch.cat([cat_emb_flat, num_x], dim=1)  # [batch, num_cat * embed_dim + num_num]
        deep_out = self.deep_mlp(deep_input).squeeze(-1)  # [batch]

        # ========== 合并输出 ==========
        logits = first_order + second_order + deep_out
        return logits

    def predict_proba(self, cat_x: Dict[str, torch.Tensor], num_x: torch.Tensor) -> torch.Tensor:
        """返回购买概率 [0, 1]，用于推理。"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(cat_x, num_x)
            return torch.sigmoid(logits)
