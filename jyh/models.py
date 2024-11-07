import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerSelf(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, nhead, num_layers):
        super(TransformerSelf, self).__init__()
        self.user_emb = torch.nn.Embedding(num_users, embedding_dim)
        self.item_emb = torch.nn.Embedding(num_items, embedding_dim)

        encoder_layers = TransformerEncoderLayer(embedding_dim, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.fc = torch.nn.Linear(embedding_dim, 1)

    def forward(self, mini_batch):
        user = mini_batch['user']
        pos_item = mini_batch['pos_item']
        neg_item = mini_batch['neg_item']

        # 获取用户和物品的嵌入
        u = self.user_emb(user)  # .unsqueeze(1)  # (batch_size, 1, embedding_dim)
        pos_emb = self.item_emb(pos_item)  # .unsqueeze(1)  # (batch_size, 1, embedding_dim)
        neg_emb = self.item_emb(neg_item)  # .unsqueeze(1)  # (batch_size, 1, embedding_dim)

        # 合并嵌入
        pos_input = torch.cat([u, pos_emb], dim=1)  # (batch_size, 2, embedding_dim)
        neg_input = torch.cat([u, neg_emb], dim=1)  # (batch_size, 2, embedding_dim)

        # Transformer编码
        pos_output = self.transformer_encoder(pos_input)  # (batch_size, 2, embedding_dim)
        neg_output = self.transformer_encoder(neg_input)  # (batch_size, 2, embedding_dim)

        # 取最后一个位置的输出
        pos_score = self.fc(pos_output[:, -1, :])  # (batch_size, 1)
        neg_score = self.fc(neg_output[:, -1, :])  # (batch_size, 1)

        return pos_score, neg_score, torch.cat((pos_emb, neg_emb), dim=-1), u

    def get_embedding_weights(self):
        user = self.user_emb.weight
        item = self.item_emb.weight

        return user, item
    def get_embedding(self, mini_batch):
        user = mini_batch['user']
        pos_item = mini_batch['pos_item']
        neg_item = mini_batch['neg_item']

        # 获取用户和物品的嵌入
        u = self.user_emb(user)  # .unsqueeze(1)  # (batch_size, 1, embedding_dim)
        pos_emb = self.item_emb(pos_item)  # .unsqueeze(1)  # (batch_size, 1, embedding_dim)
        neg_emb = self.item_emb(neg_item)  # .unsqueeze(1)  # (batch_size, 1, embedding_dim)
        return torch.cat((pos_emb, neg_emb), dim=-1), u
    def get_score_mat(self, user=None):
        u_emb, i_emb = self.get_embedding_weights()
        mat = u_emb @ i_emb.T
        if user is None:
            return mat
        else:
            return torch.index_select(mat, 0, user)
    
    def get_top_k(self, K, user=None):
        score_mat = self.get_score_mat()
        if user is None:
            topk = torch.topk(score_mat, K, dim=1)
        else:
            topk = torch.topk(torch.index_select(score_mat, 0, user), K, dim=1)
        return topk