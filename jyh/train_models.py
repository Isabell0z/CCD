import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from models import TransformerSelf, GCNSelf, VAESelf, MFSelf
from dataset import KDDataset

model_names = ["tranformer", "GCN", "VAE", "MF"]
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = "teachers"
model_name = model_names[3]
# 加载数据集
yelp_data = "original-CCD/dataset/Yelp/TASK_0.pickle"
# csv_file = 'Data/user_item_pairs.csv'
# num_users = 100
# num_items = 1000
num_negatives = 1

print(f"loading dataset")
with open(yelp_data, "rb") as f:
    data = pickle.load(f)
train_dataset = KDDataset(pickle_data=data, level='train')
dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

# 示例参数
embedding_dim = 1024
user_num = train_dataset.get_user_num()
item_num = train_dataset.get_item_num()
# 初始化模型和优化器
# model = TransformerSelf(num_users=user_num, num_items=item_num, embedding_dim=embedding_dim, nhead=4, num_layers=2)
# model = GCNSelf(num_users=user_num, num_items=item_num, hidden_channels=embedding_dim)
model = MFSelf(num_users=user_num, num_items=item_num, embedding_dim=embedding_dim)
# model = VAESelf(num_users=user_num, num_items=item_num, embedding_dim=embedding_dim)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"training")
# 训练模型
model.train()
pbar = tqdm(range(1000), desc="training epoch")
for epoch in pbar:
    total_loss = 0
    for batch in dataloader:
        batch = {k: batch[k].to(device) for k in batch.keys()}
        optimizer.zero_grad()

        pos_score, neg_score, item_emb, user_emb = model(batch)
        loss = -torch.sum(torch.log(torch.sigmoid(pos_score - neg_score)))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    pbar.set_postfix({"loss": total_loss / len(dataloader)})
    # print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')
if not os.path.exists(save_path):
    os.makedirs(save_path)
torch.save({"checkpoint": model.state_dict(),
            "score_mat": model.get_score_mat(),
            "sorted_mat": model.get_top_k(1000)[1]
            }, f"{save_path}/{model_name}.pt")
print("Training complete.")
