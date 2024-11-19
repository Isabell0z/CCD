import pickle
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import sys
from tqdm import tqdm
from self_models.BaseModels import TransformerSelf, GCNSelf, VAESelf, MFSelf
from KD_utils.dataset import KDDataset


def main(args):
    torch.manual_seed(1)
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    save_path = f"ckpts/{args.dataset}/teachers"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load datas
    print(f"loading dataset")
    data_pickle = f"dataset/{args.dataset}/TASK_0.pickle"
    with open(data_pickle, "rb") as f:
        data = pickle.load(f)
    train_dataset = KDDataset(pickle_data=data, level="train")
    dataloader = DataLoader(train_dataset, batch_size=1024 * 2, shuffle=True)

    # hyper params
    embedding_dim = 512
    user_num = train_dataset.get_user_num()
    item_num = train_dataset.get_item_num()
    # init models and optims
    model_name = args.model
    if model_name == "TransformerSelf":
        model = TransformerSelf(
            num_users=user_num,
            num_items=item_num,
            embedding_dim=embedding_dim,
            nhead=4,
            num_layers=2,
        )
    elif model_name == "GCNSelf":
        model = GCNSelf(
            num_users=user_num, num_items=item_num, hidden_channels=embedding_dim
        )
    elif model_name == "MFSelf":
        model = MFSelf(
            num_users=user_num, num_items=item_num, embedding_dim=embedding_dim
        )
    elif model_name == "VAESelf":
        model = VAESelf(
            num_users=user_num, num_items=item_num, embedding_dim=embedding_dim
        )
    else:
        print("invalid model type, QUIT")
        sys.exit()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00132)

    print(f"training")
    # train
    model.train()
    pbar = tqdm(range(args.max_epoch), desc="training epoch")
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
    if not os.path.exists(f"{save_path}/{model_name}"):
        os.makedirs(f"{save_path}/{model_name}")
    torch.save(
        {
            "checkpoint": model.state_dict(),
            "score_mat": model.get_score_mat(),
            "sorted_mat": model.get_top_k(1000).indices,
        },
        f"{save_path}/{model_name}/TASK_0.pth",
    )
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, help="TransformerSelf, GCNSelf, MFSelf or VAESelf"
    )
    parser.add_argument(
        "--dataset", "--d", type=str, default=None, help="Gowalla or Yelp"
    )
    parser.add_argument("--cuda", "--c", type=str, default="0", help="device id")
    parser.add_argument("--max_epoch", "--me", type=int)
    # parser.add_argument("--id", "-i", type=int, default=0, help="model id")
    args = parser.parse_args()
    main(args)
