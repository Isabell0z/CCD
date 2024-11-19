import argparse
import torch


def ensemble(args):
    models_names = [
        f"ckpts/{args.dataset}/teachers/{args.model}/TASK_0_{i}.pth" for i in range(5)
    ]
    m0_ckpt = torch.load(models_names[0])
    score_mat = m0_ckpt["score_mat"].unsqueeze(0)
    # sorted_mat = m0_ckpt["sorted_mat"]
    for i in range(1, 5):
        mi_ckpt = torch.load(models_names[i])
        score_mat_i = mi_ckpt["score_mat"].unsqueeze(0)
        # sorted_mat_i = mi_ckpt["sorted_mat"]
        score_mat = torch.cat((score_mat, score_mat_i), 0)
        print(i)
        # sorted_mat = torch.stack(sorted_mat, sorted_mat_i)
    score_mat_mean = torch.mean(score_mat, dim=0)
    sorted_mat = torch.topk(score_mat_mean, k=1000, dim=1).indices
    save_path = f"ckpts/{args.dataset}/teachers/{args.model}/TASK_0.pth"
    torch.save({"score_mat": score_mat_mean, "sorted_mat": sorted_mat}, save_path)
    # sorted_mat_mean = torch.mean(sorted_mat, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, help="TransformerSelf, GCNSelf, MFSelf or VAESelf"
    )
    parser.add_argument(
        "--dataset", "--d", type=str, default=None, help="Gowalla or Yelp"
    )
    args = parser.parse_args()
    ensemble(args)
