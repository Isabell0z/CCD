import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_self(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels):
        super(GCN_self, self).__init__()
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.item_emb = torch.nn.Embedding(num_items, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, mini_batch):
        '''
            mini_batch = {
                'user': torch.tensor([0, 1, 2]),  # 用户索引
                'pos_item': torch.tensor([1, 2, 3]),  # 正样本物品索引
                'neg_item': torch.tensor([3, 1, 0])   # 负样本物品索引
            }
        '''
        user = mini_batch['user']
        pos_item = mini_batch['pos_item']
        neg_item = mini_batch['neg_item']

        # 获取用户和物品的嵌入
        user_emb = self.user_emb(user)
        pos_item_emb = self.item_emb(pos_item)
        neg_item_emb = self.item_emb(neg_item)

        # 构建图数据
        num_users = user_emb.size(0)
        num_pos_items = pos_item_emb.size(0)
        num_neg_items = neg_item_emb.size(0)
        total_nodes = num_users + num_pos_items + num_neg_items

        # 调整 edge_index 的构建
        user_indices = torch.arange(num_users)
        pos_item_indices = torch.arange(num_pos_items) + num_users
        neg_item_indices = torch.arange(num_neg_items) + num_users + num_pos_items

        edge_index = torch.cat([user_indices.unsqueeze(0), pos_item_indices.unsqueeze(0)], dim=0)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 双向边

        # 检查 edge_index 的范围
        if edge_index.max() >= total_nodes:
            raise ValueError(f"edge_index contains invalid indices: {edge_index.max()} >= {total_nodes}")

        x = torch.cat([user_emb, pos_item_emb, neg_item_emb], dim=0)

        # # 打印调试信息
        # print(f"edge_index: {edge_index}")
        # print(f"x shape: {x.shape}")

        # 图卷积操作
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # 计算正负样本评分
        u = x[:num_users]
        i = x[num_users:num_users + num_pos_items]
        j = x[num_users + num_pos_items:]

        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        return pos_score, neg_score

    
    