import argparse
import time
import gc
import sys
import os
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Utils.data_loaders import *
from Utils.utils import *


def main(args):
    """Main function for training and evaluation in Stage1:Student generation"""

    # Set up GPU and scaler
    gpu = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    scaler = torch.cuda.amp.GradScaler()
    print(f"GPU = {gpu}")

    # Extract model type and random seed
    model_type = args.model  # e.g.,
    model_seed = 0
    # Set random Seed
    print(f"Random_seed = {model_seed}")
    set_random_seed(int(model_seed))

    # Load the path of student-side models (S_proxy, P_proxy, Student)
    Student_load_path = f"ckpts/{args.dataset}/students/{args.model}/Test"
    load_S_proxy_dir_path = f"{Student_load_path}/Stability"
    load_P_proxy_dir_path = f"{Student_load_path}/Plasticity"
    load_CL_model_dir_path = f"{Student_load_path}/CL"

    # Load the path of teacher
    teacher_dir = f""
    print("Teacher_load_path", teacher_dir)

    # Load dataset
    data_path = f"dataset/{args.dataset}/total_blocks_timestamp.pickle"
    data_dict_path = f"dataset/{args.dataset}"

    total_blocks = load_pickle(data_path)
    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = (
        load_data_as_dict(data_dict_path, num_task=args.num_task)
    )

    # Determine target task index
    distillation_idx = args.target_task
    print(f"KD task: ")

    # Data block for Stage 1 (i.e., k-1 th)
    p_block = total_blocks[distillation_idx]
    p_total_user = p_block.user.max() + 1
    p_total_item = p_block.item.max() + 1
    p_train_dict = total_train_dataset[f"TASK_{distillation_idx}"]
    p_train_interaction, p_train_mat, p_valid_mat, p_test_mat = (
        get_train_valid_test_mat(
            distillation_idx,
            total_train_dataset,
            total_valid_dataset,
            total_test_dataset,
        )
    )
    R = make_R(p_total_user, p_total_item, p_train_mat)  # Rating matrix

    # load the teacher model
    teacher_path = (
        f"ckpts/{args.dataset}/teachers/{args.model}/TASK_{distillation_idx}.pth"
    )
    t_ckpt = torch.load(teacher_path, map_location=gpu)
    T_score_mat = t_ckpt["score_mat"].detach().cpu()
    T_sorted_mat = t_ckpt["sorted_mat"].detach().cpu()

    # lfilter new datas
    if distillation_idx >= 1:
        b_block = total_blocks[distillation_idx - 1]
        b_total_user = b_block.user.max() + 1
        new_user_train_mat, new_user_valid_mat, new_user_test_mat = (
            get_train_valid_test_mat_for_new_users(
                b_total_user, p_total_user, p_train_mat, p_valid_mat, p_test_mat
            )
        )

        print(f"loaded new data")
        new_user_results = get_eval_with_mat(
            new_user_train_mat,
            new_user_valid_mat,
            new_user_test_mat,
            T_sorted_mat,
            args.k_list,
        )
        print(
            f"\tvalid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}"
        )

    # Get negatvie exclude data
    negatvie_exclude = get_negative_exclude(
        distillation_idx,
        p_train_dict,
        p_total_user,
        gpu,
        load_S_proxy_dir_path,
        load_P_proxy_dir_path,
        load_CL_model_dir_path,
    )

    # Get dataset
    RRD_item_ids = torch.arange(p_total_item)
    RRD_train_dataset, IR_reg_train_dataset, BPR_train_dataset = get_RRD_IR_BPR_dataset(
        T_score_mat,
        negatvie_exclude,
        p_train_dict,
        p_train_mat,
        p_train_interaction,
        p_total_user,
        p_total_item,
        args.nns,
        args.nui,
        args.nuu,
    )
    train_loader = DataLoader(
        deepcopy(BPR_train_dataset), batch_size=args.bs, shuffle=True, drop_last=False
    )

    # Get gpu memory
    del negatvie_exclude, T_score_mat, T_sorted_mat
    gc.collect()
    torch.cuda.empty_cache()

    # Get model
    p_SNM = get_SNM(p_total_user, p_total_item, R, gpu)
    D_Student = get_model(
        p_total_user, p_total_item, p_SNM, gpu, args, model_type, model_weight=None
    ).to(gpu)

    print(f"loaded model")

    optimizer = optim.Adam(D_Student.parameters(), lr=args.lr, weight_decay=1e-4)
    eval_args = {
        "best_score": 0,
        "test_score": 0,
        "best_epoch": 0,
        "best_model": None,
        "score_mat": None,
        "sorted_mat": None,
        "patience": 0,
        "avg_valid_score": 0,
        "avg_test_score": 0,
    }

    total_time = 0.0
    for epoch in range(args.max_epoch):
        print(f"\n[KD_Epoch : {epoch + 1} / {args.max_epoch}]")
        start_time = time.time()
        train_loader.dataset.negative_sampling()

        if epoch % args.uninterested_sample_epoch == 0:
            RRD_train_dataset.sampling_for_uninteresting_items()
            IR_reg_train_dataset.sampling_for_uninteresting_users()

        epoch_URRD_loss = 0.0
        epoch_IR_RRD_loss = 0.0
        epoch_CF_loss = 0.0

        # CF + RRD
        D_Student.train()
        for mini_batch in train_loader:
            mini_batch = {key: values.to(gpu) for key, values in mini_batch.items()}
            batch_user = mini_batch["user"]
            batch_pos_item = mini_batch["pos_item"]
            batch_neg_item = mini_batch["neg_item"]

            ## CF ##
            user_emb, item_emb = D_Student.base_model.get_embedding_weights()

            batch_user_emb = user_emb[batch_user]
            batch_pos_item_emb = item_emb[batch_pos_item]
            batch_neg_item_emb = item_emb[batch_neg_item]

            pos_score = (batch_user_emb * batch_pos_item_emb).sum(dim=1, keepdim=True)
            neg_score = (batch_user_emb * batch_neg_item_emb).sum(dim=1, keepdim=True)

            base_loss = (pos_score - neg_score).sigmoid()
            base_loss = torch.clamp(base_loss, min=0.0001, max=0.9999)
            base_loss = -base_loss.log().sum()
            epoch_CF_loss += base_loss.item()

            ## RRD ##
            batch_user = batch_user.unique()
            interesting_items, uninteresting_items = RRD_train_dataset.get_samples(
                batch_user.cpu()
            )

            interesting_items = interesting_items.to(gpu)
            uninteresting_items = uninteresting_items.to(gpu)

            interesting_prediction = forward_multi_items(
                user_emb, item_emb, batch_user, interesting_items
            )
            uninteresting_prediction = forward_multi_items(
                user_emb, item_emb, batch_user, uninteresting_items
            )

            URRD_loss = args.URRD_lambda * relaxed_ranking_loss(
                interesting_prediction, uninteresting_prediction
            )
            epoch_URRD_loss += URRD_loss.item()
            batch_loss = base_loss + URRD_loss

            # backward (Cf + RRD)
            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        ## IR_RRD ##
        iteration = (len(RRD_item_ids) // args.bs) + 1
        shuffle_item_ids = RRD_item_ids[torch.randperm(RRD_item_ids.size(0))]
        for idx in range(iteration):

            if idx + 1 == iteration:
                batch_item = shuffle_item_ids[idx * args.bs :]
            else:
                batch_item = shuffle_item_ids[idx * args.bs : (idx + 1) * args.bs]

            user_emb, item_emb = D_Student.base_model.get_embedding_weights()
            interesting_users, uninteresting_users = IR_reg_train_dataset.get_samples(
                batch_item
            )

            batch_item = batch_item.to(gpu)
            interesting_users = interesting_users.to(gpu)
            uninteresting_users = uninteresting_users.to(gpu)

            interesting_user_prediction = forward_multi_users(
                user_emb, item_emb, interesting_users, batch_item
            )
            uninteresting_user_prediction = forward_multi_users(
                user_emb, item_emb, uninteresting_users, batch_item
            )

            IR_reg = relaxed_ranking_loss(
                interesting_user_prediction, uninteresting_user_prediction
            )
            IR_RRD_loss = args.IR_reg_lmbda * IR_reg
            epoch_IR_RRD_loss += IR_RRD_loss.item()

            # backward (IR_RRD)
            optimizer.zero_grad()
            scaler.scale(IR_RRD_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Training Result
        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time

        epoch_CF_loss = round(epoch_CF_loss / len(train_loader), 4)
        epoch_URRD_loss = round(epoch_URRD_loss / len(train_loader), 4)
        epoch_IR_RRD_loss = round(epoch_IR_RRD_loss / iteration, 4)

        print(
            f"epoch_CF_loss = {epoch_CF_loss}, epoch_URRD_loss = {epoch_URRD_loss}, epoch_IR_RRD_loss = {epoch_IR_RRD_loss}",
            end=" ",
        )
        print(
            f"epoch_time = {epoch_time:.4f} seconds, total_time = {total_time:.4f} seconds"
        )

        # Evaluation Result
        if epoch % args.eval_cycle == 0:
            print("\n[Evaluation]")
            D_Student.eval()
            with torch.no_grad():
                D_score_mat, D_sorted_mat = get_sorted_score_mat(
                    D_Student, topk=1000, return_sorted_mat=True
                )

            valid_list, test_list = get_CL_result(
                total_train_dataset,
                total_valid_dataset,
                total_test_dataset,
                D_sorted_mat,
                args.k_list,
                current_task_idx=distillation_idx,
                FB_flag=False,
                return_value=True,
            )

            avg_valid_score = get_average_score(
                valid_list[: distillation_idx + 1], "valid_R20"
            )
            avg_test_score = get_average_score(
                test_list[: distillation_idx + 1], "test_R20"
            )

            if args.eval_average_acc:
                valid_score = avg_valid_score
                test_score = avg_test_score
            else:
                valid_score = valid_list[distillation_idx]["valid_R20"]
                test_score = test_list[distillation_idx]["test_R20"]

            if distillation_idx >= 1:
                new_user_results = get_eval_with_mat(
                    new_user_train_mat,
                    new_user_valid_mat,
                    new_user_test_mat,
                    D_sorted_mat,
                    args.k_list,
                )
                new_user_results_txt = f"valid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}"
            else:
                new_user_results_txt = None

            if valid_score > eval_args["best_score"]:
                print(
                    f"\t[Best Model Changed]\n\tvalid_score = {valid_score:.4f}, test_score = {test_score:.4f}"
                )
                eval_args["best_score"] = valid_score
                eval_args["test_score"] = test_score
                eval_args["avg_valid_score"] = avg_valid_score
                eval_args["avg_test_score"] = avg_test_score

                eval_args["best_epoch"] = epoch
                eval_args["best_model"] = deepcopy(
                    {k: v.cpu() for k, v in D_Student.state_dict().items()}
                )  # D_Student
                eval_args["score_mat"] = deepcopy(D_score_mat)
                eval_args["sorted_mat"] = deepcopy(D_sorted_mat)
                eval_args["patience"] = 0

                best_new_user_results = deepcopy(new_user_results_txt)
            else:
                eval_args["patience"] += 1

        # get gpu memory
        gc.collect()
        torch.cuda.empty_cache()

    # Final Result Report
    print(f"\n[Final result of KD (Stage1) in the distilled idx {distillation_idx}]")
    print(
        f"best_epoch = {eval_args['best_epoch']}, valid_score = {eval_args['best_score']}, test_score = {eval_args['test_score']}"
    )
    print(
        f"avg_valid_score = {eval_args['avg_valid_score']}, avg_test_score = {eval_args['avg_test_score']}"
    )

    get_CL_result(
        total_train_dataset,
        total_valid_dataset,
        total_test_dataset,
        eval_args["sorted_mat"],
        args.k_list,
        current_task_idx=args.num_task,
        FB_flag=False,
        return_value=False,
    )

    if distillation_idx >= 1:
        print(f"\n\t[The Result of new users in {distillation_idx}-th Block]")
        print(f"\t{best_new_user_results}\n")

    # Paramters save
    if args.save:

        save_path = Student_load_path
        save_dir_path = os.path.join(save_path, "Distilled")
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        save_path = os.path.join(save_dir_path, f"TASK_{distillation_idx}.pth")
        model_state = {
            "best_model": eval_args["best_model"],  # .cpu(),
            "score_mat": eval_args["score_mat"],
            "sorted_mat": torch.topk(eval_args["score_mat"], k=1000, dim=1).indices,
        }

        torch.save(model_state, save_path)

        print("saved model at path = ", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--uninterested_sample_epoch", type=int, default=10)

    # UR-RRD
    parser.add_argument("--URRD_lambda", "--ul", type=float, default=0.01)
    parser.add_argument(
        "--nii", help="the number of interesting items", type=int, default=40
    )
    parser.add_argument(
        "--nui", help="the number of uninteresting items", type=int, default=10000
    )  # 10000)

    # IR-RRD
    parser.add_argument("--IR_reg_lmbda", "--il", type=float, default=0.01)

    # LWCKD + PIW
    parser.add_argument("--nc", help="num_cluster", type=int, default=10)
    parser.add_argument("--T", help="temperature", type=int, default=5.0)
    parser.add_argument(
        "--LWCKD_flag",
        "--lf",
        action=argparse.BooleanOptionalAction,
        help="whether using LWC_KD or not (--lf or --no-lf)",
    )
    parser.add_argument(
        "--PIW_flag",
        "--pf",
        action=argparse.BooleanOptionalAction,
        help="whether using PIW or not (--pf or --no-pf)",
    )
    parser.add_argument("--LWCKD_lambda", type=float, default=0.01)
    parser.add_argument("--cluster_lambda", type=float, default=1)

    # setup
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bs", help="batch_size", type=int, default=1024)
    parser.add_argument(
        "--nns", help="the number of negative sample", type=int, default=1
    )
    parser.add_argument("--sd", help="student_dims", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--reg", type=float, default=0.0001)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--eval_cycle", type=int, default=5)
    parser.add_argument(
        "--eval_average_acc",
        "--eaa",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--eaa or --no-eaa)",
    )
    parser.add_argument("--k_list", type=list, default=[20, 50, 100])
    parser.add_argument("--num_task", type=int, default=6)
    parser.add_argument(
        "--random_init",
        "--r",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--r or --no-r)",
    )

    # SAVE
    parser.add_argument(
        "--save",
        "--s",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--s or --no-s)",
    )

    # Data / Model / Target_task (i.e., k-th data block)
    parser.add_argument(
        "--dataset", "--d", type=str, default=None, help="Gowalla or Yelp"
    )
    parser.add_argument(
        "--model", "-m", type=str, help="TransformerSelf, GCN, MF or VAE"
    )
    parser.add_argument("--target_task", "--tt", type=int, default=-1)

    args = parser.parse_args()

    # Arguments for list-wise distllation and the dimension for student.
    if args.dataset == "Gowalla":
        args.nui = 30000  # the number of unintersted items (nui)
        args.nuu = 19500  # the number of unintersted users (nuu)
        if args.sd == None:
            args.sd = 16

    elif args.dataset == "Yelp":
        args.nui = 10000
        args.nuu = 11500
        if args.sd == None:
            args.sd = 8

    print_command_args(args)
    main(args)
    print_command_args(args)

# run_code: python KD.py --d Yelp -m LightGCN_1 --tt 1 --max_epoch 10
