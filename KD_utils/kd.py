import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import os

from KD_utils.DE import Expert, compute_DE_loss
from KD_utils.RRD import compute_RRD_loss
from KD_utils.dataset import KDDataset
from self_models.BaseModels import TransformerSelf, VAESelf, GCNSelf, MFSelf
from Utils.utils import merge_model_kd, eval_task


class KD:
    def __init__(
        self,
        dataloader,
        experts_num,
        experts_dims,
        teacher_model,
        student_model: nn.Module,
        K,
        experts=None,
        DE_lambda=0.001,
        RRD_lambda=0.0001,
        device="cuda",
    ):
        """_summary_

        Args:
            dataloader (Dataloader):
            experts_num (int): number of experts
            experts_dims (list): [in, hidden, out];
                    in=student_embedding_size, out=teacher_embedding_size
            teacher_model (nn.Module): teacher model
            student_model (nn.Module): student model
            K (int): length of recommandation list
            DE_lambda (float, optional): Defaults to 0.01.
            RRD_lambda (float, optional): Defaults to 0.001.
            device (str, optional): Defaults to 'cuda'.
        """
        self.dataloader = dataloader
        self.experts_num = experts_num
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device

        self.user_experts = nn.ModuleList(
            [
                Expert(experts_dims[0], experts_dims[1], experts_dims[2])
                for _ in range(self.experts_num)
            ]
        )
        # pos + neg -> size*2
        self.item_experts = nn.ModuleList(
            [
                Expert(experts_dims[0] * 2, experts_dims[1] * 2, experts_dims[2] * 2)
                for _ in range(self.experts_num)
            ]
        )
        if experts is not None:
            self.user_experts.load_state_dict(experts["user_experts"])
            self.item_experts.load_state_dict(experts["item_experts"])
        self.user_experts.to(device)
        self.item_experts.to(device)
        self.K = K
        self.DE_lambda = DE_lambda
        self.RRD_lambda = RRD_lambda

    def train(self, epoch):
        self.teacher_model.eval()
        self.student_model.train()
        optimizer = torch.optim.Adam(
            [
                {"params": self.user_experts.parameters()},
                {"params": self.item_experts.parameters()},
            ],
            lr=0.001,
        )
        optimizer_base = torch.optim.Adam(self.student_model.parameters(), lr=0.0005)
        pbar = tqdm(range(epoch), desc="Training")
        t_score_mat = self.teacher_model.get_score_mat()
        for e in pbar:
            loss_sum = 0.0
            base_loss_sum = 0.0
            for i, batch in enumerate(self.dataloader):
                optimizer.zero_grad()
                optimizer_base.zero_grad()
                batch = {k: batch[k].to(self.device) for k in batch.keys()}
                with torch.no_grad():
                    t_item_emb, t_user_emb = self.teacher_model.get_embedding(batch)
                s_pos_score, s_neg_score, s_item_emb, s_user_emb = self.student_model(
                    batch
                )
                # DE loss
                expert_idx = e % self.experts_num
                user_expert = self.user_experts[expert_idx]
                item_expert = self.item_experts[expert_idx]
                user_DE_loss = compute_DE_loss(s_user_emb, t_user_emb, user_expert)
                item_DE_loss = compute_DE_loss(s_item_emb, t_item_emb, item_expert)

                # RRD loss
                if len(batch["user"].shape) > 1:
                    u = batch["user"].squeeze(1)
                else:
                    u = batch["user"]
                pred_t = torch.index_select(t_score_mat, 0, u)
                pred_s = self.student_model.get_score_mat(u)
                RRD_loss = compute_RRD_loss(pred_t, pred_s, self.K)

                # Base loss
                base_loss = -torch.sum(
                    torch.log(torch.sigmoid(s_pos_score - s_neg_score))
                )

                total_loss = (
                    self.DE_lambda * (user_DE_loss + item_DE_loss)
                    + self.RRD_lambda * RRD_loss
                    + base_loss
                )
                # total_loss = base_loss
                total_loss.backward()

                optimizer.step()
                optimizer_base.step()
                loss_sum += total_loss.item()
                base_loss_sum += base_loss.item()
            pbar.set_postfix({"loss": loss_sum, "base_loss": base_loss_sum})

    def eval(self, task, args):
        sorted_mat = self.student_model.get_top_k(K=20)
        task_data = f"dataset/{args.dataset}/TASK_{task}.pickle"
        r_mean = eval_task(task_data, sorted_mat.indices, k=20, save_txt=True)
        print(r_mean)

    def save_models(self, save_path, task, loaded_model=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if loaded_model is not None:
            best_model = merge_model_kd(loaded_model, self.student_model.state_dict())
        else:
            best_model = None
        torch.save(
            {
                "best_model": best_model,
                "checkpoint": self.student_model.state_dict(),
                "score_mat": self.student_model.get_score_mat(),
                "sorted_mat": self.student_model.get_top_k(1000)[1],
            },
            os.path.join(save_path, f"TASK_{task}.pth"),
        )
        torch.save(
            self.item_experts.state_dict(),
            os.path.join(save_path, f"{self.experts_num}_item_experts.pth"),
        )
        torch.save(
            self.user_experts.state_dict(),
            os.path.join(save_path, f"{self.experts_num}_user_experts.pth"),
        )


def main(args):
    # set device
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    # load data
    print(f"loading dataset")
    pickle_data = f"dataset/{args.dataset}/TASK_{args.target_task}.pickle"
    with open(pickle_data, "rb") as f:
        data = pickle.load(f)
    students_path = f"ckpts/{args.dataset}/students"
    model_name = args.model
    # if not os.path.exists(f"{students_path}/{model_name}"):
    #     os.makedirs(f"{students_path}/{model_name}_0")
    train_dataset = KDDataset(pickle_data=data, level="train")
    dataloader = DataLoader(train_dataset, batch_size=2**12, shuffle=True)
    # init teacher model
    teacher_embedding_dim = 512
    student_embedding_dim = 8
    user_num = train_dataset.get_user_num()
    item_num = train_dataset.get_item_num()

    # init or load student model
    if model_name == "TransformerSelf":
        teacher_model = TransformerSelf(
            num_users=user_num,
            num_items=item_num,
            embedding_dim=teacher_embedding_dim,
            nhead=4,
            num_layers=2,
        )
        student_model = TransformerSelf(
            num_users=user_num,
            num_items=item_num,
            embedding_dim=student_embedding_dim,
            nhead=4,
            num_layers=2,
        )
    elif model_name == "GCNSelf":
        teacher_model = GCNSelf(
            num_users=user_num,
            num_items=item_num,
            hidden_channels=teacher_embedding_dim,
        )
        student_model = GCNSelf(
            num_users=user_num,
            num_items=item_num,
            hidden_channels=student_embedding_dim,
        )
    elif model_name == "MFSelf":
        teacher_model = MFSelf(
            num_users=user_num, num_items=item_num, embedding_dim=teacher_embedding_dim
        )
        student_model = MFSelf(
            num_users=user_num, num_items=item_num, embedding_dim=student_embedding_dim
        )
    elif model_name == "VAESelf":
        teacher_model = VAESelf(
            num_users=user_num, num_items=item_num, embedding_dim=teacher_embedding_dim
        )
        student_model = VAESelf(
            num_users=user_num, num_items=item_num, embedding_dim=student_embedding_dim
        )
    else:
        print("invalid model type, QUIT")
        sys.exit()

    teacher_path = (
        f"ckpts/{args.dataset}/teachers/{args.model}/TASK_{args.target_task}.pth"
    )
    ckpt = torch.load(teacher_path)
    teacher_model.load_state_dict(ckpt["checkpoint"])
    if args.target_task > 0:
        s_ckpt = torch.load(
            f"ckpts/{args.dataset}/students/{args.model}/Test/CL/TASK_{args.target_task}.pth"
        )
        student_model.load_state_dict(s_ckpt["checkpoint"])
    else:
        s_ckpt = None
    if args.load:
        experts = {
            "item_experts": torch.load(
                f"{students_path}/{model_name}_{args.target_task-1}/5_item_experts.pth"
            ),
            "user_experts": torch.load(
                f"{students_path}/{model_name}_{args.target_task-1}/5_user_experts.pth"
            ),
        }
    else:
        experts = None
    # KD
    # instantiate kd
    kd = KD(
        dataloader=dataloader,
        experts_num=5,
        experts_dims=[
            student_embedding_dim,
            teacher_embedding_dim * 2,
            teacher_embedding_dim,
        ],
        teacher_model=teacher_model,
        student_model=student_model,
        K=20,
        experts=experts,
        device=device,
    )
    # run kd
    kd.train(epoch=args.max_epoch)
    kd.eval(args.target_task, args)
    # save
    if s_ckpt is not None:
        kd.save_models(
            f"{students_path}/{model_name}/Test/Distilled/",
            task=args.target_task,
            loaded_model=s_ckpt["best_model"],
        )
    else:
        kd.save_models(
            f"{students_path}/{model_name}/Test/Distilled/",
            task=args.target_task,
            loaded_model=None,
        )
    # end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, help="TransformerSelf, GCNSelf, MFSelf or VAESelf"
    )
    parser.add_argument(
        "--dataset", "--d", type=str, default=None, help="Gowalla or Yelp"
    )
    parser.add_argument("--cuda", "-c", type=str, default="0", help="device id")
    parser.add_argument(
        "--load",
        type=bool,
        default=False,
        help="Whether load pre-trained student model",
    )
    parser.add_argument("--target_task", "--tt", type=int, help="target task id")
    parser.add_argument("--max_epoch", "--me", type=int, default=100)
    args = parser.parse_args()
    main(args)
