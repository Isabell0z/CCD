import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import trange
random.seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_data(sum_dict):
    data = []
    for user in sum_dict.keys():
        for item in sum_dict[user]:
            data.append([user, item])
    return data


class KDDataset(Dataset):
    def __init__(self, pickle_data, level='train'):
        if level == 'train':
            pos_items = pickle_data['train_dict']
        elif level == 'val':
            pos_items = pickle_data['valid_dict']
        else:
            pos_items = pickle_data['test_dict']
        self.data = parse_data(pos_items)
        self.pos_items = pos_items
        self.num_users = pickle_data['num_base_block_users']
        self.num_items = pickle_data['num_base_block_items']
        self.neg_items = {}

        for i in trange(self.num_users):
            self.neg_items[i] = []
            for j in range(self.num_items):
                if j not in self.pos_items[i]:
                    self.neg_items[i].append(j)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        pos_item = self.data[idx][1]
        neg_item = random.choice(self.neg_items[user])
        mini_batch = {"user": torch.LongTensor([user]),
                      "pos_item": torch.LongTensor([pos_item]),
                      "neg_item": torch.LongTensor([neg_item])}
        return mini_batch

    def get_user_num(self):
        return self.num_users

    def get_item_num(self):
        return self.num_items
