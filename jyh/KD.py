import torch
import torch.nn as nn
from tqdm import tqdm

from DE import Expert, compute_DE_loss
from RRD import compute_RRD_loss


class KD:
    def __init__(self, dataloader, experts_num, experts_dims, t_model, s_model, K, DE_lambda=0.01, RRD_lambda=0.001,
                 device='cuda'):
        self.dataloader = dataloader
        self.experts_num = experts_num
        self.t_model = t_model
        self.s_model = s_model
        self.experts = nn.ModuleList([Expert(experts_dims[0], experts_dims[1], experts_dims[2])])
        self.K = K
        self.DE_lambda = DE_lambda
        self.RRD_lambda = RRD_lambda

    def train(self, epoch):
        self.t_model.eval()
        self.s_model.train()
        optimizer = torch.optim.Adam([{"params": self.s_model.parameters()}, {"params": self.experts.parameters()}],
                                     lr=0.01)
        pbar = tqdm(range(epoch), desc="epoch")
        for e in pbar:
            loss_sum = 0.0

            for i, batch in enumerate(self.dataloader):
                optimizer.zero_grad()

                with torch.no_grad():
                    t_pos_score, t_neg_score, t_item_emb, t_user_emb = self.t_model(batch)
                s_pos_score, s_neg_score, s_item_emb, s_user_emb = self.s_model(batch)
                # DE loss
                expert = self.experts[e % self.experts_num]
                user_DE_loss = compute_DE_loss(s_user_emb, t_user_emb, expert)
                item_DE_loss = compute_DE_loss(s_item_emb, t_item_emb, expert)

                # RRD loss
                pred_t = t_pos_score - t_neg_score
                pred_s = s_pos_score - s_neg_score
                RRD_loss = compute_RRD_loss(pred_t, pred_s, self.K)

                # Base loss
                base_loss = -torch.mean(torch.log(torch.sigmoid(s_pos_score - s_neg_score)))

                total_loss = self.DE_lambda * (user_DE_loss + item_DE_loss) \
                             + self.RRD_lambda * RRD_loss \
                             + base_loss
                total_loss.backward()

                optimizer.step()
                loss_sum += total_loss.item()
            pbar.set_postfix({"loss": loss_sum})


if __name__ == "__main__":
    # init teacher models and student models

    # load data

    # KD

    # save

    # end
    pass


