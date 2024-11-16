import argparse
import time
import random
import time
import gc
import sys
import os
import ipdb
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Utils.data_loaders import *
from Utils.utils import *


# Student update:

# goal:
#######
# next student model
# update proxy and using proxy-guided replay learning
#######


def main(args):
    # preliminary : Load data
    #######
    # current data ID: current_task
    # before data block: before_user, before_item
    # current data block: current_user, current_item
    # new user and item interaction: new_user, new_item
    # current student model: Distill_Student_model
    # current proxy: S_proxy, P_proxy
    ######

    # set GPU
    gpu = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    # scaler = torch.cuda.amp.GradScaler()
    print(f"GPU = {gpu}")

    # load data

    # get current and distillation id
    current_task = args.target_task
    before_task = current_task - 1
    print(f"\n Current task = {current_task}, Distillation task = {before_task}")

    # get new user and item number
    data_block_path = f"dataset/{args.dataset}/total_blocks_timestamp.pickle"
    total_blocks = load_pickle(data_block_path)
    before_block = total_blocks[before_task]
    current_block = total_blocks[current_task]
    before_user = before_block.user.max() + 1
    before_item = before_block.item.max() + 1
    current_user = current_block.user.max() + 1
    current_item = current_block.item.max() + 1
    new_user = current_user - before_user
    new_item = current_item - before_item
    print(f"\n before user ={before_user}, current user ={current_user}")
    print(f"\n before item ={before_item}, current item ={current_item}")
    print(f"\n new user ={new_user}, new item ={new_item}")

    # load distillation train data, all current train, valid, test data
    data_dict_path = f"dataset/{args.dataset}"
    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = (
        load_data_as_dict(data_dict_path, num_task=args.num_task)
    )
    before_train_data = total_train_dataset[f"TASK_{before_task}"]
    current_train_data = total_train_dataset[f"TASK_{current_task}"]
    current_valid_data = total_valid_dataset[f"TASK_{current_task}"]
    current_test_data = total_test_dataset[f"TASK_{current_task}"]

    # get new user train, valid, test rating matrix, before train rating matrix
    before_train_mat = make_rating_mat(before_train_data)
    current_train_mat = make_rating_mat(current_train_data)
    current_valid_mat = make_rating_mat(current_valid_data)
    current_test_mat = make_rating_mat(current_test_data)
    new_user_train, new_user_valid, new_user_test = (
        get_train_valid_test_mat_for_new_users(
            before_user,
            current_user,
            current_train_mat,
            current_valid_mat,
            current_test_mat,
        )
    )

    # set student model and seed
    model_type, _, model_seed = args.model.split("_")
    set_random_seed(int(model_seed))
    print(f"\n Model = {model_type}, Random seed = {model_seed}")

    # load student model

    Distill_Student_rootpath = (
        f"ckpts/{args.dataset}/students/{model_type}/Test/Distilled"
    )
    Distill_Student_path = os.path.join(
        Distill_Student_rootpath, f"TASK_{before_task}.pth"
    )
    Distill_model_weight, Distill_score_mat, Distill_sorted_mat = load_saved_model(
        Distill_Student_path, gpu
    )
    before_R = make_R(before_user, before_item, before_train_mat)  # get score matrix
    before_SNM = get_SNM(before_user, before_item, before_R, gpu)
    base_model_only = True if model_seed == "0" else False

    Distill_Student_model = get_model(
        before_user,
        before_item,
        before_SNM,
        gpu,
        args,
        model_type,
        Distill_model_weight,
        base_model_only=base_model_only,
    ).to(gpu)
    print(f"\n The distillation student model={Distill_Student_model}")
    del before_R, before_SNM

    # load P proxy, S proxy
    if before_task == 0:
        # do not have S proxy and P proxy before, get from distillation model
        P_proxy = deepcopy(Distill_Student_model)
        S_proxy = deepcopy(Distill_Student_model)
        P_score_mat = deepcopy(Distill_score_mat)
        S_score_mat = deepcopy(Distill_score_mat)
    else:
        # according to the weight of P proxy and S proxy to create P proxy and S proxy model
        P_proxy_rootpath = f"ckpts/{args.dataset}/students/{model_type}/Test/Plasticity"
        S_proxy_rootpath = f"ckpts/{args.dataset}/students/{model_type}/Test/Stability"
        P_proxy_path = os.path.join(P_proxy_rootpath, f"TASK_{before_task-1}.pth")
        S_proxy_path = os.path.join(S_proxy_rootpath, f"TASK_{before_task-1}.pth")
        P_proxy_weight = torch.load(P_proxy_path)["best_model"]
        S_proxy_weight = torch.load(S_proxy_path)["best_model"]
        proxy_get_data = before_task - 1
        proxy_get_block = total_blocks[proxy_get_data]
        proxy_train_data = total_train_dataset[f"TASK_{proxy_get_data}"]
        proxy_get_user = proxy_get_block.user.max() + 1
        proxy_get_item = proxy_get_block.item.max() + 1
        proxy_train_mat = make_rating_mat(proxy_train_data)
        proxy_R = make_R(proxy_get_user, proxy_get_item, proxy_train_mat)
        proxy_SNM = get_SNM(proxy_get_user, proxy_get_item, proxy_R, gpu)
        P_proxy = get_model(
            proxy_get_user,
            proxy_get_item,
            proxy_SNM,
            gpu,
            args,
            model_type,
            P_proxy_weight,
        )
        S_proxy = get_model(
            proxy_get_user,
            proxy_get_item,
            proxy_SNM,
            gpu,
            args,
            model_type,
            S_proxy_weight,
        )
        del proxy_R, proxy_SNM

        # update proxy using current distillation data
        S_proxy = merge_model(
            S_proxy,
            Distill_Student_model,
            wme=True,
            b_weight=args.s_weight,
            p_weight=1 - args.s_weight,
        ).to(gpu)
        P_proxy = merge_model(
            P_proxy,
            Distill_Student_model,
            wme=True,
            b_weight=args.p_weight,
            p_weight=1 - args.p_weight,
        ).to(gpu)
        S_proxy = freeze(S_proxy)
        P_proxy = freeze(P_proxy)
        P_score_mat, P_sorted_mat = get_sorted_score_mat(
            P_proxy, return_sorted_mat=True
        )
        S_score_mat, S_sorted_mat = get_sorted_score_mat(
            S_proxy, return_sorted_mat=True
        )

    print(f"\n P proxy={P_proxy}")
    print(f"\n P proxy score matrix={P_score_mat}")
    print(f"\n S proxy={S_proxy}")
    print(f"\n S proxy score matrix={P_score_mat}")

    #################### eval
    # test the performance of current distillation student model, P proxy, S proxy

    # use function get_CL_result

    #################### eval

    # step1 : Learning with new interactions
    ######
    # entity embedding initialization technique
    # using 2-hop relations: items purchased together, users who bought the same item
    ######

    # get top 40 items from Teacher, Student, S and P proxy

    Teacher_path = f"ckpts/{args.dataset}/teachers/{model_type}/TASK_{before_task}.pth"
    # else:
    #     Teacher_path = f"ckpts/{args.dataset}/teachers/using_student_{model_type}/TASK_{before_task}_score_mat.pth"
    Teacher_score_mat = (
        torch.load(Teacher_path, map_location=gpu)["score_mat"].detach().cpu()
    )
    Filter_Teacher_score_mat = (
        filtering_simple(Teacher_score_mat, before_train_data).detach().cpu()
    )
    T_top_items = torch.topk(Filter_Teacher_score_mat, k=40, dim=1).indices
    exclude_top_items = T_top_items.clone().detach()

    get_CL_result(
        total_train_dataset,
        total_valid_dataset,
        total_test_dataset,
        Distill_sorted_mat,
        args.k_list,
        current_task_idx=args.num_task - 1,
        FB_flag=False,
        return_value=False,
        max_task=current_task,
    )
    exclude_top_items = torch.cat(
        [exclude_top_items, torch.tensor(Distill_sorted_mat[:, :40])], dim=1
    )
    del Distill_sorted_mat

    if before_task != 0:
        get_CL_result(
            total_train_dataset,
            total_valid_dataset,
            total_test_dataset,
            S_sorted_mat,
            args.k_list,
            current_task_idx=current_task,
            FB_flag=False,
            return_value=False,
            max_task=args.num_task - 1,
        )
        exclude_top_items = torch.cat(
            [exclude_top_items, torch.tensor(S_sorted_mat[:, :40])], dim=1
        )
        get_CL_result(
            total_train_dataset,
            total_valid_dataset,
            total_test_dataset,
            P_sorted_mat,
            args.k_list,
            current_task_idx=current_task,
            FB_flag=False,
            return_value=False,
            max_task=args.num_task - 1,
        )
        exclude_top_items = torch.cat(
            [exclude_top_items, torch.tensor(P_sorted_mat[:, :40])], dim=1
        )
        del S_sorted_mat, P_sorted_mat

    # get BPR train dataset
    current_interaction = make_interaction(current_train_data)
    BPR_train_data = implicit_CF_dataset(
        before_user,
        before_item,
        before_train_mat,
        args.nns,
        current_interaction,
        exclude_top_items,
    )
    train_data = DataLoader(
        BPR_train_data, batch_size=args.bs, shuffle=True, drop_last=False
    )

    # initialize model
    S_model = deepcopy(Distill_Student_model).to(gpu)
    for param in S_model.parameters():
        param.requires_grad = True

    # update new item and user
    before_user_mapping, before_item_mapping, before_rating_mat, UU, II = (
        None,
        None,
        None,
        None,
        None,
    )
    before_user_ids, before_item_ids, present_user_ids, present_item_ids = (
        None,
        None,
        None,
        None,
    )
    R = make_R(current_user, current_item, current_train_mat)
    SNM = get_SNM(current_user, current_item, R, gpu)
    S_model.update(
        before_user_ids,
        before_item_ids,
        before_user_mapping,
        before_item_mapping,
        before_rating_mat,
        new_user,
        new_item,
        UU,
        II,
        present_user_ids,
        present_item_ids,
        R,
        args.random_init,
        SNM,
        args.init_topk,
        args.only_one_hop,
    )

    # step2 : Proxy-guided replay learning initialize

    # filter S score mat using before and current train data
    Filter_S_score_mat = filtering_simple(S_score_mat, before_train_data)
    Filter_S_score_mat = filtering_simple(Filter_S_score_mat, current_train_data)

    # filter P score mat using before and current train data
    if before_task == 0:
        Filter_P_score_mat = Filter_S_score_mat
    else:
        Filter_P_score_mat = filtering_simple(P_score_mat, before_train_mat)
        Filter_P_score_mat = filtering_simple(Filter_P_score_mat, current_train_data)

    if args.replay_learning:
        if args.Using_S:
            S_rank_mat = convert_to_rank_mat(Filter_S_score_mat)
            S_sigmoid_mat = torch.sigmoid(Filter_S_score_mat)
        else:
            S_rank_mat = None
            S_sigmoid_mat = None

        if args.Using_P:
            if before_task != 0 or (before_task == 0 and not args.Using_S):
                P_rank_mat = convert_to_rank_mat(Filter_P_score_mat)
                P_sigmoid_mat = torch.sigmoid(Filter_P_score_mat)
            else:
                P_rank_mat = None
                P_sigmoid_mat = None

        if before_task == 0 and args.Using_P and args.Using_S != True:
            args.P_sample = args.S_sample

        replay_learning_data = get_total_replay_learning_dataset(
            Distill_score_mat,
            S_rank_mat,
            S_sigmoid_mat,
            P_rank_mat,
            P_sigmoid_mat,
            args,
        )
        print(f"\n replay learing initialized")
        print(f"\n S_rank_mat:{S_rank_mat}")
        print(f"\n P_rank_mat:{P_rank_mat}")
        print(f"\n S_sigmoid_mat:{S_sigmoid_mat}")
        print(f"\n P_sigmoid_mat:{P_sigmoid_mat}")

    # step3 : train model (replay learning)  and the overall objective for student update

    # initialize model
    if model_type == "LightGCN":
        S_model.base_model.set_layer_index(args.using_layer_index)
        S_model.base_model.num_layer = args.num_layer

    optimizer = optim.Adam(S_model.parameters(), lr=args.lr, weight_decay=args.reg)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_time = 0.0

    # train model
    for epoch in range(args.max_epoch):
        # initialize
        print(f"\n[Epoch:{epoch + 1}/{args.max_epoch}]")
        start_time = time.time()
        train_data.dataset.negative_sampling()

        # train (continue learning)
        CF_loss = 0.0
        S_model.train()
        for batch in train_data:
            batch = {key: values.to(gpu) for key, values in batch.items()}
            embedding = S_model.base_model.forward(batch)

            # updata model
            batch_loss = S_model.base_model.get_loss(embedding[0], embedding[1])
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # scaler.scale(batch_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            CF_loss += batch_loss.item()

        CF_time = time.time()
        print(
            f"epoch_CF_loss = {round(CF_loss / len(train_data), 4)}, CF_time = {CF_time - start_time:.4f} seconds",
            end=" ",
        )

        # train (replay learning)
        if args.replay_learning:
            if args.annealing:
                replay_learning_lambda = args.replay_learning_lambda * torch.exp(
                    torch.tensor(-epoch) / args.a_T
                )
            else:
                replay_learning_lambda = args.replay_learning_lambda

            # shuffle (may delete???)
            shuffled_list = random.sample(
                replay_learning_data, len(replay_learning_data)
            )
            user_list, item_list, rate_list = list(zip(*shuffled_list))
            iteration = (len(shuffled_list) // args.bs) + 1

            replay_learning_loss = 0.0

            for id in range(iteration):

                # replay learning start and end
                if id + 1 == iteration:
                    start, end = id * args.bs, -1
                else:
                    start, end = id * args.bs, (id + 1) * args.bs

                # Batch
                batch_user = torch.tensor(user_list[start:end]).to(gpu)
                batch_item = torch.tensor(item_list[start:end]).to(gpu)
                batch_label = torch.tensor(rate_list[start:end]).to(gpu)

                user_embedding, item_embedding = (
                    S_model.base_model.get_embedding_weights()
                )
                batch_user_emb = user_embedding[batch_user]
                batch_item_emb = item_embedding[batch_item]

                # update
                output = (batch_user_emb * batch_item_emb).sum(1)
                batch_loss = criterion(output, batch_label)
                batch_loss *= replay_learning_lambda

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                # scaler.scale(batch_loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                replay_learning_loss += batch_loss.item()

            replay_learning_time = time.time()
            print(
                f"epoch_replay_learning_loss = {round(replay_learning_loss / iteration, 4)}, replay_learning_lambda = {replay_learning_lambda:.5f}, replay_learning_time = {replay_learning_time - CF_time:.4f} seconds",
                end=" ",
            )

            # Time
            end_time = time.time()
            epoch_time = end_time - start_time
            total_time += epoch_time
            print(
                f"epoch_time = {epoch_time:.4f} seconds, total_time = {total_time:.4f} seconds"
            )

            #################### eval
            # evaluation model

            # save best model

            #################### eval

    # save model
    if args.save:
        print("save model")
        print("\n save P,S proxy and CL model")
        with torch.no_grad():
            CL_score_mat, CL_sorted_mat = get_sorted_score_mat(
                S_model, return_sorted_mat=True
            )
        if args.save_path is not None:
            save_path = args.save_path
        else:
            save_path = f"ckpts/{args.dataset}/students/{model_type}/Test"

        save_S_proxy_dir_path = f"{save_path}/Stability"
        save_P_proxy_dir_path = f"{save_path}/Plasticity"
        save_CL_dir_path = f"{save_path}/CL"

        for dir_path in [save_S_proxy_dir_path, save_P_proxy_dir_path]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        save_model(
            save_S_proxy_dir_path,
            before_task,
            {
                "best_model": deepcopy(
                    {k: v.cpu() for k, v in S_proxy.state_dict().items()}
                ),
                "score_mat": Filter_S_score_mat,
            },
        )
        save_model(
            save_P_proxy_dir_path,
            before_task,
            {
                "best_model": deepcopy(
                    {k: v.cpu() for k, v in P_proxy.state_dict().items()}
                ),
                "score_mat": Filter_P_score_mat,
            },
        )
        save_model(
            save_CL_dir_path,
            current_task,
            {
                "best_model": deepcopy(
                    {k: v.cpu() for k, v in S_model.state_dict().items()}
                ),
                "score_mat": CL_score_mat,
            },
        )

        print("save_S_proxy_dir_path", save_S_proxy_dir_path)
        print("save_P_proxy_dir_path", save_P_proxy_dir_path)
        print("save_CL_dir_model", save_CL_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save",
        "--s",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--s or --no-s)",
    )
    parser.add_argument("--save_path", "--sp", type=str)

    # etc
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=0.0001)
    parser.add_argument("--bs", help="batch_size", type=int, default=2048)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--rs", type=int, default=1, help="random seed")
    parser.add_argument("--early_stop", type=int, default=2)
    parser.add_argument(
        "--nns", help="the number of negative sample", type=int, default=1
    )
    parser.add_argument("--sd", help="student_dims", type=int, default=8)

    # S/P proxies & Replay Learning
    parser.add_argument(
        "--replay_learning",
        "--rl",
        action=argparse.BooleanOptionalAction,
        help="whether using param or not (--replay_learning or --no-replay_learning)",
    )
    parser.add_argument("--eps", type=float, default=0.0001)
    parser.add_argument("--a_T", type=float, default=10.0)
    parser.add_argument("--replay_learning_lambda", type=float, default=0.5)
    parser.add_argument("--s_weight", "--sw", type=float, default=0.9)
    parser.add_argument("--p_weight", "--pw", type=float, default=0.9)
    parser.add_argument("--absolute", "--ab", type=int, default=100)
    parser.add_argument(
        "--S_sample",
        "--ss",
        type=int,
        default=5,
        help="hyper parameter for Ranking Distillation",
    )
    parser.add_argument(
        "--P_sample",
        "--ps",
        type=int,
        default=5,
        help="hyper parameter for Ranking Distillation",
    )
    parser.add_argument(
        "--annealing", "--a", action="store_true", default=True, help="using annealing"
    )

    # Initaltation for new users/items
    parser.add_argument(
        "--random_init",
        "--r",
        action="store_true",
        default=False,
        help="random_initalization for new user/items",
    )
    parser.add_argument(
        "--init_topk", type=int, default=20
    )  # how many popular neighbors you would like to aggregate
    parser.add_argument(
        "--only_one_hop",
        "--ooh",
        action=argparse.BooleanOptionalAction,
        help="whether using param or not (--ooh or --no-ooh)",
    )

    # Evalation
    parser.add_argument("--num_task", type=int, default=6)
    parser.add_argument(
        "--hyper_param",
        "--h",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--h or --no-h)",
    )
    parser.add_argument(
        "--eval_average_acc",
        "--eaa",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--eaa or --no-eaa)",
    )
    parser.add_argument("--k_list", type=list, default=[20, 50, 100])

    # LightGCN
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument(
        "--using_layer_index", type=str, default="avg", help="first, last, avg"
    )

    # PIW
    parser.add_argument("--nc", help="num_cluster", type=int, default=10)
    parser.add_argument("--T", help="temperature", type=int, default=5.0)

    # Data / Model / Target_task (i.e., k-th data block)
    parser.add_argument(
        "--model", "-m", type=str, help="TransformerSelf, GCNSelf, MFSelf or VAESelf"
    )
    parser.add_argument(
        "--dataset", "--d", type=str, default=None, help="Gowalla or Yelp"
    )
    parser.add_argument("--target_task", "--tt", type=int, default=-1)

    # Toggle for S/P proxies, and Distilled Student
    parser.add_argument(
        "--Using_S",
        "--US",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--s or --no-s)",
    )
    parser.add_argument(
        "--Using_P",
        "--UP",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--s or --no-s)",
    )
    parser.add_argument(
        "--Using_D",
        "--UD",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--s or --no-s)",
    )

    args = parser.parse_args()

    # Student's embedding dimension
    if args.sd == None:
        if args.dataset == "Gowalla":
            args.sd = 16

        elif args.dataset == "Yelp":
            args.sd = 8

    print_command_args(args)
    main(args)
    print_command_args(args)
