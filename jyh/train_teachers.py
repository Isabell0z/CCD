import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from models import TransformerSelf
from dataset import KDDataset



# 加载数据集
yelp_data = r"D:\Projects\CCD\original-CCD\dataset\Yelp\TASK_0.pickle"
# csv_file = 'Data/user_item_pairs.csv'
# num_users = 100
# num_items = 1000
num_negatives = 1

print(f"loading dataset")
with open(yelp_data, "rb") as f:
    data = pickle.load(f)
train_dataset = KDDataset(pickle_data=data, level='train')
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 示例参数
embedding_dim = 8
user_num = train_dataset.get_user_num()
item_num = train_dataset.get_item_num()
# 初始化模型和优化器
model = TransformerSelf(num_users=user_num, num_items=item_num, embedding_dim=embedding_dim, nhead=4, num_layers=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"training")
# 训练模型
model.train()
pbar = tqdm(range(10), desc="training epoch")
for epoch in pbar:
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        pos_score, neg_score = model(batch)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    pbar.set_postfix({"loss": total_loss / len(dataloader)})
    # print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')
torch.save(model.state_dict(), f"teachers/transformer.pt")
print("Training complete.")
