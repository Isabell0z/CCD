import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import os

from DE import Expert, compute_DE_loss
from RRD import compute_RRD_loss
from dataset import KDDataset
from models import TransformerSelf

device = "cuda" if torch.cuda.is_available() else "cpu"


class KD:
    def __init__(
        self,
        dataloader,
        experts_num,
        experts_dims,
        t_model,
        s_model: nn.Module,
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
            t_model (nn.Module): teacher model
            s_model (nn.Module): student model
            K (int): length of recommandation list
            DE_lambda (float, optional): Defaults to 0.01.
            RRD_lambda (float, optional): Defaults to 0.001.
            device (str, optional): Defaults to 'cuda'.
        """
        self.dataloader = dataloader
        self.experts_num = experts_num
        self.t_model = t_model.to(device)
        self.s_model = s_model.to(device)

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
        self.t_model.eval()
        self.s_model.train()
        optimizer = torch.optim.Adam(
            [
                {"params": self.user_experts.parameters()},
                {"params": self.item_experts.parameters()},
            ],
            lr=0.01,
        )
        optimizer_base = torch.optim.Adam(self.s_model.parameters(), lr=0.0005)
        pbar = tqdm(range(epoch), desc="Training")
        t_score_mat = self.t_model.get_score_mat()
        for e in pbar:
            loss_sum = 0.0
            base_loss_sum = 0.0
            for i, batch in enumerate(self.dataloader):
                optimizer.zero_grad()
                optimizer_base.zero_grad()
                batch = {k: batch[k].to(device) for k in batch.keys()}
                with torch.no_grad():
                    t_item_emb, t_user_emb = self.t_model.get_embedding(batch)
                s_pos_score, s_neg_score, s_item_emb, s_user_emb = self.s_model(batch)
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
                pred_s = student_model.get_score_mat(u)
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

    def save_models(self, save_path):
        torch.save(
            {
                "checkpoint": self.s_model.state_dict(),
                "score_mat": self.s_model.get_score_mat(),
                "sorted_mat": self.s_model.get_topk(1000)[1],
            },
            os.path.join(save_path, "task0_student.pth"),
        )
        torch.save(
            self.item_experts.state_dict(),
            os.path.join(save_path, f"{self.experts_num}_item_experts.pth"),
        )
        torch.save(
            self.user_experts.state_dict(),
            os.path.join(save_path, f"{self.experts_num}_user_experts.pth"),
        )


if __name__ == "__main__":
    # load data
    print(f"loading dataset")
    yelp_data = "original-CCD/dataset/Yelp/TASK_0.pickle"
    with open(yelp_data, "rb") as f:
        data = pickle.load(f)
    students_path = "students"
    model_name = "transformer"
    if not os.path.exists(f"{students_path}/{model_name}"):
        os.makedirs(f"{students_path}/{model_name}")
    train_dataset = KDDataset(pickle_data=data, level="train")
    dataloader = DataLoader(train_dataset, batch_size=2**12, shuffle=True)
    # init teacher model
    teacher_embedding_dim = 1024
    student_embedding_dim = 128
    user_num = train_dataset.get_user_num()
    item_num = train_dataset.get_item_num()
    teacher_model = TransformerSelf(
        num_users=user_num,
        num_items=item_num,
        embedding_dim=teacher_embedding_dim,
        nhead=4,
        num_layers=2,
    )
    ckpt = torch.load("teachers/task0_transformer.pth")
    teacher_model.load_state_dict(ckpt["checkpoint"])
    # init or load student model
    student_model = TransformerSelf(
        num_users=user_num,
        num_items=item_num,
        embedding_dim=student_embedding_dim,
        nhead=2,
        num_layers=2,
    )
    ckpt_s = torch.load(f"{students_path}/{model_name}/task0_student.pt")
    student_model.load_state_dict(ckpt_s)
    # load expert models
    experts = {
        "item_experts": torch.load(f"{students_path}/{model_name}/5_item_experts.pt"),
        "user_experts": torch.load(f"{students_path}/{model_name}/5_user_experts.pt"),
    }
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
        t_model=teacher_model,
        s_model=student_model,
        K=10,
        experts=experts,
    )
    # run kd
    kd.train(epoch=10)
    # save
    kd.save_models(f"{students_path}/{model_name}")
    # end
