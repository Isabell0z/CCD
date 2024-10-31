import torch
import torch.nn.functional as F

class MF_self(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MF_self, self).__init__()
        self.user_emb = torch.nn.Embedding(num_users, embedding_dim)
        self.item_emb = torch.nn.Embedding(num_items, embedding_dim)

    def forward(self, mini_batch):
        user = mini_batch['user']
        pos_item = mini_batch['pos_item']
        neg_item = mini_batch['neg_item']

        # 获取用户和物品的嵌入
        u = self.user_emb(user)
        i = self.item_emb(pos_item)
        j = self.item_emb(neg_item)

        # 计算正负样本评分
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        return pos_score, neg_score
