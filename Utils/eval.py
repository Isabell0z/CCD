import pickle
import numpy as np
import torch
from Utils.utils import eval_task, eval_all_task, h_mean, make_rating_mat


def eval(max_task, args, la_list, k=20):
    ckpts = torch.load(
        f"ckpts/{args.dataset}/students/Test/Distilled/TASK_{max_task}.pth"
    )
    score_mat = ckpts["score_mat"]
    sorted_mat = torch.topk(score_mat, k, dim=1)
    ra_list = eval_all_task(sorted_mat, max_task, args, k)
    print("RA:", ra_list)
    h_mean_list = []
    for i in range(max_task):
        h_mean_list.append(h_mean(la_list[i], ra_list[i]))
    print("H-mean:", h_mean_list)


def ndcg(task_data, sorted_mat, k=20):
    with open(task_data, "rb") as f:
        task_data = pickle.load(f)
    sorted_mat = sorted_mat.cpu().numpy()
    gt_mat = make_rating_mat(task_data["test_dict"])
    denom = np.log2(np.arange(2, k + 2))
    ndcg_list = []
    for u in task_data["test_dict"].keys():
        test_user = u
        dcg_k = np.sum(
            np.in1d(sorted_mat[test_user, :k], list(gt_mat[test_user].keys())) / denom
        )
        idcg_k = np.sum((1 / denom)[: min(len(list(gt_mat[test_user].keys())), k)])
        NDCG_k = dcg_k / idcg_k
        ndcg_list.append(NDCG_k)
    ndcg_ = np.mean(np.array(ndcg_list))
    return ndcg_


def eval_teacher():
    task_data = "dataset/Yelp/TASK_0.pickle"
    ckpt = torch.load("ckpts/TASK_0_score_mat.pth")
    score_mat = ckpt["score_mat"]
    sorted_mat = torch.topk(score_mat, 20, dim=1).indices
    # r = eval_task(task_data, sorted_mat, save_txt=False)
    ndcg_ = ndcg(task_data, sorted_mat)
    print(ndcg_)


if __name__ == "__main__":
    eval_teacher()
