import argparse
import torch
from self_models.BaseModels import MFSelf, TransformerSelf


def fix_single_model(model_name, model_type="MFSelf"):
    mi_ckpt = torch.load(model_name)
    if model_type == "MFSelf":
        mi = MFSelf(12248, 10822, embedding_dim=512)
    elif model_type == "TransformerSelf":
        mi = TransformerSelf(12248, 10822, 512, 4, 2)
    mi.load_state_dict(mi_ckpt["checkpoint"])
    mi.to("cuda")
    score_mat = mi.get_score_mat()
    sorted_mat = mi.get_top_k(1000).indices
    save_path = model_name
    torch.save(
        {
            "checkpoint": mi.state_dict(),
            "score_mat": score_mat,
            "sorted_mat": sorted_mat,
        },
        save_path,
    )


def fix(args):
    models_names = [
        f"ckpts/{args.dataset}/teachers/{args.model}/TASK_0_{i}.pth" for i in range(5)
    ]

    # sorted_mat = m0_ckpt["sorted_mat"]
    for i in range(0, 5):
        fix_single_model(models_names[i])
        print(i)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--model", "-m", type=str, help="TransformerSelf, GCNSelf, MFSelf or VAESelf"
    # )
    # parser.add_argument(
    #     "--dataset", "--d", type=str, default=None, help="Gowalla or Yelp"
    # )
    # args = parser.parse_args()
    # fix(args)
    fix_single_model(
        "ckpts/Yelp/teachers/TransformerSelf/Task_0.pth", "TransformerSelf"
    )
