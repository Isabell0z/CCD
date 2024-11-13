import torch
import pickle
from torch.utils.data import DataLoader
from dataset import KDDataset
from models import TransformerSelf


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
embedding_dim = 128
user_num = train_dataset.get_user_num()
item_num = train_dataset.get_item_num()
model = TransformerSelf(num_users=user_num, num_items=item_num, embedding_dim=embedding_dim, nhead=4, num_layers=2)
# ckpt = torch.load("teachers/transformer.pt")
ckpt = torch.load("students/transformer/student.pt")
model.load_state_dict(ckpt)
score_mat = model.get_score_mat()
sorted_mat = model.get_top_k(1000)
state_dict = {
    "checkpoint": model.state_dict(),
    "score_mat": score_mat,
    "sorted_mat": sorted_mat[1]
}
torch.save(state_dict, "students/transformer/task0_student.pth")
