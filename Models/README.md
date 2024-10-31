# CCD_Model

Model inputs:

```
mini_batch = {
    'user': torch.tensor([0, 1, 2]),  # 用户索引
    'pos_item': torch.tensor([1, 2, 3]),  # 正样本物品索引
    'neg_item': torch.tensor([3, 1, 0])   # 负样本物品索引
}

```
Model outputs:
```
pos_score, neg_score
```
Model Reference:
```
from Models.GCN_self import GCN_self
from Models.MF_self import MF_self
from Models.Transformer_self import Transformer_self
from Models.VAE_self import VAE_self

#Teacher
embedding_dim = 128
#Student
embedding_dim = 8

model = MF_self(num_users, num_items, embedding_dim)
model = GCN_self(num_users, num_items, embedding_dim)
model = Transformer_self(num_users, num_items, embedding_dim, nhead=4, num_layers=2)
model = VAE_self(num_users, num_items, embedding_dim)


```