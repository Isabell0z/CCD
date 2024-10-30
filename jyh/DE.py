import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, s_dim, hidden_dim, t_dim):
        """

        Args:
            s_dim: dimension of student domain
            hidden_dim: dimension of hidden layer
            t_dim: dimension of teacher domain
        """
        super(Expert, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(s_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, t_dim),
                                 )

    def forward(self, x):
        return self.mlp(x)


class Selection(nn.Module):
    def __init__(self, t_dim, num_experts):
        super(Selection, self).__init__()
        self.mlp = nn.Linear(t_dim, num_experts)

    def forward(self, x):
        logits = self.mlp(x)
        return nn.functional.gumbel_softmax(logits, tau=0.01, dim=1)


def compute_DE_loss(s_emb, t_emb, expert):
    # compute s -> t
    expert_result = expert(s_emb)

    # compute loss
    DE_loss = nn.functional.mse_loss(expert_result, t_emb, reduction='mean')
    return DE_loss
