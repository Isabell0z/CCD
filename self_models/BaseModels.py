import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ModelSelf(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(ModelSelf, self).__init__()
        self.user_count = num_users
        self.item_count = num_items
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

    def forward(self, mini_batch):
        pass

    def get_embedding_weights(self):
        user = self.user_emb.weight
        item = self.item_emb.weight

        return user, item

    def get_embedding(self, mini_batch):
        user = mini_batch["user"]
        pos_item = mini_batch["pos_item"]
        neg_item = mini_batch["neg_item"]

        # get embeddings of users and items
        u = self.user_emb(user)  # .unsqueeze(1)  # (batch_size, 1, embedding_dim)
        pos_emb = self.item_emb(
            pos_item
        )  # .unsqueeze(1)  # (batch_size, 1, embedding_dim)
        neg_emb = self.item_emb(
            neg_item
        )  # .unsqueeze(1)  # (batch_size, 1, embedding_dim)
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

    def get_loss(self, pos_score, neg_score):
        return -torch.sum(torch.log(torch.sigmoid(pos_score - neg_score)))


class TransformerSelf(ModelSelf):
    def __init__(self, num_users, num_items, embedding_dim, nhead, num_layers):
        super(TransformerSelf, self).__init__(num_users, num_items, embedding_dim)
        # self.user_emb = nn.Embedding(num_users, embedding_dim)
        # self.item_emb = nn.Embedding(num_items, embedding_dim)

        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, mini_batch):
        """_summary_

        Args:
            mini_batch (dict): {"user":[], "pos_item":[], "neg_item":[]}

        Returns:
            list: pos score, neg score, items embedding(pos, neg), users embedding
        """
        user = mini_batch["user"]
        pos_item = mini_batch["pos_item"]
        neg_item = mini_batch["neg_item"]

        # 获取用户和物品的嵌入

        u = self.user_emb(user)  # (batch_size, 1, embedding_dim)
        pos_emb = self.item_emb(pos_item)  # (batch_size, 1, embedding_dim)
        neg_emb = self.item_emb(neg_item)  # (batch_size, 1, embedding_dim)
        if len(u.shape) < len(pos_emb.shape):
            u = u.unsqueeze(1)
        if len(u.shape) == 2:
            u = u.unsqueeze(1)
            pos_emb = pos_emb.unsqueeze(1)
            neg_emb = neg_emb.unsqueeze(1)
        # 合并嵌入
        # print("SHAPE SHAPE SHAPE", u.shape, pos_emb.shape)
        pos_input = torch.cat([u, pos_emb], dim=1)  # (batch_size, 2, embedding_dim)
        neg_input = torch.cat([u, neg_emb], dim=1)  # (batch_size, 2, embedding_dim)

        # Transformer编码
        # print("size:", user.shape, u.shape, pos_input.shape)
        pos_output = self.transformer_encoder(
            pos_input
        )  # (batch_size, 2, embedding_dim)
        neg_output = self.transformer_encoder(
            neg_input
        )  # (batch_size, 2, embedding_dim)

        # 取最后一个位置的输出
        pos_score = self.fc(pos_output[:, -1, :])  # (batch_size, 1)
        neg_score = self.fc(neg_output[:, -1, :])  # (batch_size, 1)

        return pos_score, neg_score, torch.cat((pos_emb, neg_emb), dim=-1), u


class GCNSelf(ModelSelf):
    def __init__(self, num_users, num_items, hidden_channels):
        super(GCNSelf, self).__init__(
            num_users, num_items, embedding_dim=hidden_channels
        )
        # self.user_emb = nn.Embedding(num_users, hidden_channels)
        # self.item_emb = nn.Embedding(num_items, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, mini_batch):
        """
        mini_batch = {
            'user': torch.tensor([0, 1, 2]),  # 用户索引
            'pos_item': torch.tensor([1, 2, 3]),  # 正样本物品索引
            'neg_item': torch.tensor([3, 1, 0])   # 负样本物品索引
        }
        """
        user = mini_batch["user"]
        pos_item = mini_batch["pos_item"]
        neg_item = mini_batch["neg_item"]

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

        edge_index = torch.cat(
            [user_indices.unsqueeze(0), pos_item_indices.unsqueeze(0)], dim=0
        )
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 双向边

        # 检查 edge_index 的范围
        if edge_index.max() >= total_nodes:
            raise ValueError(
                f"edge_index contains invalid indices: {edge_index.max()} >= {total_nodes}"
            )

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
        pos_emb = x[num_users : num_users + num_pos_items]
        neg_emb = x[num_users + num_pos_items :]

        pos_score = (u * pos_emb).sum(dim=1, keepdim=True)
        neg_score = (u * neg_emb).sum(dim=1, keepdim=True)

        return pos_score, neg_score, torch.cat((pos_emb, neg_emb), dim=-1), u


class MFSelf(ModelSelf):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MFSelf, self).__init__(num_users, num_items, embedding_dim)
        # self.user_emb = nn.Embedding(num_users, embedding_dim)
        # self.item_emb = nn.Embedding(num_items, embedding_dim)

    def forward(self, mini_batch):
        user = mini_batch["user"]
        pos_item = mini_batch["pos_item"]
        neg_item = mini_batch["neg_item"]

        # 获取用户和物品的嵌入
        u = self.user_emb(user)
        pos_emb = self.item_emb(pos_item)
        neg_emb = self.item_emb(neg_item)

        # 计算正负样本评分
        pos_score = (u * pos_emb).sum(dim=1, keepdim=True)
        neg_score = (u * neg_emb).sum(dim=1, keepdim=True)

        return pos_score, neg_score, torch.cat((pos_emb, neg_emb), dim=-1), u


class VAESelf(ModelSelf):
    def __init__(self, num_users, num_items, embedding_dim):
        super(VAESelf, self).__init__(num_users, num_items, embedding_dim)

        self.embedding_dim = embedding_dim

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )

        self.mu_layer = nn.Linear(64, embedding_dim)
        self.logvar_layer = nn.Linear(64, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim * 2),
        )

        self.fc = nn.Linear(embedding_dim, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, mini_batch):
        user = mini_batch["user"]
        pos_item = mini_batch["pos_item"]
        neg_item = mini_batch["neg_item"]

        # 获取用户和物品的嵌入
        u = self.user_emb(user)
        pos_emb = self.item_emb(pos_item)
        neg_emb = self.item_emb(neg_item)

        # 合并嵌入
        pos_input = torch.cat([u, pos_emb], dim=-1)
        neg_input = torch.cat([u, neg_emb], dim=-1)

        # 编码
        pos_encoded = self.encoder(pos_input)
        neg_encoded = self.encoder(neg_input)

        # 获取均值和对数方差
        pos_mu = self.mu_layer(pos_encoded)
        pos_logvar = self.logvar_layer(pos_encoded)
        neg_mu = self.mu_layer(neg_encoded)
        neg_logvar = self.logvar_layer(neg_encoded)

        # 重参数化
        pos_z = self.reparameterize(pos_mu, pos_logvar)
        neg_z = self.reparameterize(neg_mu, neg_logvar)

        # 解码
        pos_decoded = self.decoder(pos_z)
        neg_decoded = self.decoder(neg_z)

        # 计算评分
        pos_score = self.fc(pos_decoded[..., : self.embedding_dim])
        neg_score = self.fc(neg_decoded[..., : self.embedding_dim])

        return pos_score, neg_score, torch.cat((pos_emb, neg_emb), dim=-1), u
