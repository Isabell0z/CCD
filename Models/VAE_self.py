import torch
import torch.nn.functional as F
from torch import nn

class VAE_self(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(VAE_self, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(64, embedding_dim)
        self.logvar_layer = nn.Linear(64, embedding_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim * 2)
        )
        
        self.fc = nn.Linear(embedding_dim, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, mini_batch):
        user = mini_batch['user']
        pos_item = mini_batch['pos_item']
        neg_item = mini_batch['neg_item']

        # 获取用户和物品的嵌入
        u = self.user_emb(user)
        i = self.item_emb(pos_item)
        j = self.item_emb(neg_item)

        # 合并嵌入
        pos_input = torch.cat([u, i], dim=1)
        neg_input = torch.cat([u, j], dim=1)

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
        pos_score = self.fc(pos_decoded[:, :self.embedding_dim])
        neg_score = self.fc(neg_decoded[:, :self.embedding_dim])

        return pos_score, neg_score, #pos_mu, pos_logvar, neg_mu, neg_logvar
