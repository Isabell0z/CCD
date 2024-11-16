"""
Created on  Oct 28 2024
Author: REN Zhihao
Description: This script contains the function to update the teacher model.
Note: 请记得导入Utils文件夹中的data_loaders.py和utils.py
"""

import argparse
import random
import time
import gc
import sys
import os
from copy import deepcopy
import ipdb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from Utils.data_loaders import *
from Utils.utils import *

"""
    Args:
        T (torch.nn.Module): The teacher model to be updated.
        mt (str): Model type, e.g., "BPR", "LightGCN", or "VAE".
        tl (DataLoader): Training data loader.
        opt (torch.optim.Optimizer): Optimizer for the model.
        crit (torch.nn.Module): Loss function.
        sc (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        a (argparse.Namespace): Arguments containing training hyperparameters.
        ttd (Dataset): Training dataset.
        tvd (Dataset): Validation dataset.
        ttds (Dataset): Test dataset.
        Tsm (torch.Tensor): Teacher score matrix.
        Ssm (torch.Tensor): Student score matrix.
        Srm (torch.Tensor): Student replay matrix.
        Psm (torch.Tensor): Previous score matrix.
        Prm (torch.Tensor): Previous replay matrix.
        CLsm (torch.Tensor): Current learning score matrix.
        CLrm (torch.Tensor): Current learning replay matrix.
        pR (torch.Tensor): Previous replay tensor.
        ptu (torch.Tensor): Previous training updates.
        mi (int): Model index.
        ti (int): Task index.
        nu_mt (torch.Tensor): New user model tensor.
        nu_vm (torch.Tensor): New user validation matrix.
        nu_tm (torch.Tensor): New user test matrix.
        g (torch.device): Device to run the model on.
    Returns:
        dict: A dictionary containing evaluation results and best model parameters.
"""


def teacher_update(
    T,
    mt,
    tl,
    sc,
    a,
    ttd,
    tvd,
    ttds,
    Tsm,
    Ssm,
    Srm,
    Psm,
    Prm,
    CLsm,
    CLrm,
    pR,
    ptu,
    mi,
    ti,
    nu_mt,
    nu_vm,
    nu_tm,
    g,
):

    # 对于BPR或LightGCN，将模型参数和聚类参数包装在一起
    p = [{"params": T.parameters()}, {"params": T.cluster}]
    lt = ["base", "UI", "IU", "UU", "II", "cluster"]

    # 初始化Adam优化器
    opt = optim.Adam(p, lr=a.lr, weight_decay=a.reg)
    # 使用二元交叉熵loss函数
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    # 初始化评估参数
    ea = {
        "best_score": 0,
        "test_score": 0,
        "best_epoch": 0,
        "best_model": None,
        "score_mat": None,
        "sorted_mat": None,
        "base_model": None,
        "patience": 0,
        "avg_valid_score": 0,
        "avg_test_score": 0,
    }
    tt = 0

    gc.collect()
    torch.cuda.empty_cache()
    # 训练
    for e in range(a.max_epoch):
        print("\n[Epoch:" + str(e + 1) + "/" + str(a.max_epoch) + "]")
        # 初始化每种loss类型的累计值
        el = {f"epoch_{l}_loss": 0.0 for l in lt}
        erl = 0.0
        st = time.time()
        # 负采样
        tl.dataset.negative_sampling()

        T.train()
        for mb in tl:
            # 正向传播计算loss

            base_loss, UI_loss, IU_loss, UU_loss, II_loss, cluster_loss = T(mb)
            batch_loss = (
                base_loss
                + a.LWCKD_lambda * (UI_loss + IU_loss + UU_loss + II_loss)
                + (a.cluster_lambda * cluster_loss)
            )
            # 反向传播和优化
            opt.zero_grad()
            sc.scale(batch_loss).backward()
            sc.step(opt)
            sc.update()
            # 更新loss
            for l in lt:
                el[f"epoch_{l}_loss"] += eval(f"{l}_loss").item()

        for l in lt:
            ln = f"epoch_{l}_loss"
            el[ln] = round(el[ln] / len(tl), 4)

        cft = time.time()
        print(str(el) + ", CF_time = " + str(round(cft - st, 4)) + " seconds", end=" ")

        ##2. 回放学习部分
        if a.replay_learning and (mt in ["TransformerSelf"]):
            # 判断是否使用退火技术来调整回放学习的权重
            if a.annealing:
                # 使用指数退火调整lambda，随着时间的推移逐渐减少回放学习的影响
                replay_learning_lambda = a.replay_learning_lambda * torch.exp(
                    torch.tensor(-e) / a.T
                )
            else:
                replay_learning_lambda = a.replay_learning_lambda

            # 打乱回放学习数据集
            replay_learning_dataset = get_total_replay_learning_dataset_Teacher(
                Tsm, Ssm, Srm, Psm, Prm, CLsm, CLrm, a
            )
            sl = random.sample(replay_learning_dataset, len(replay_learning_dataset))
            ul, il, rl = list(zip(*sl))
            it = (len(sl) // a.bs) + 1
            erl = 0.0

            for idx in range(it):
                # 获取当前batch的数据
                if idx + 1 == it:
                    s = idx * a.bs
                    en = -1
                else:
                    s = idx * a.bs
                    en = (idx + 1) * a.bs
                # 将用户、项目和评分转换为tensor
                bu = torch.tensor(ul[s:en], dtype=torch.long).to(g)
                bi = torch.tensor(il[s:en], dtype=torch.long).to(g)
                batch_label = torch.tensor(rl[s:en], dtype=torch.float16).to(g)

                # 获取用户和项目的embedding
                ue, ie = T.base_model.get_embedding_weights()
                bue = ue[bu]
                bie = ie[bi]

                # 计算loss
                o = (bue * bie).sum(1)
                batch_loss = crit(o, batch_label)
                batch_loss *= replay_learning_lambda

                # 反向传播和优化
                opt.zero_grad()
                sc.scale(batch_loss).backward()
                sc.step(opt)
                sc.update()
                erl += batch_loss.item()

            rlt = time.time()
            print(
                "epoch_replay_learning_loss = "
                + str(round(erl / it, 4))
                + ", replay_learning_lambda = "
                + str(round(replay_learning_lambda.item(), 5))
                + ", replay_learning_time = "
                + str(round(rlt - cft, 4))
                + " seconds",
                end=" ",
            )

        et = time.time()
        etime = et - st
        tt += etime
        print(
            "epoch_time = "
            + str(round(etime, 4))
            + " seconds, total_time = "
            + str(round(tt, 4))
            + " seconds"
        )

        ##3. 评估
        if e % 5 == 0:
            print("\n[Evaluation]")
            # 设置模型为评估模式
            T.eval()
            with torch.no_grad():
                # 计算得分矩阵
                Tsm, Tsmat = get_sorted_score_mat(T, topk=1000, return_sorted_mat=True)
                # ipdb.set_trace()
            # 获取当前任务的评估结果
            vl, test_list = get_CL_result(
                ttd,
                tvd,
                ttds,
                Tsmat,
                a.k_list,
                current_task_idx=ti,
                FB_flag=False,
                return_value=True,
            )

            # 计算平均分数
            avs = get_average_score(vl[: ti + 1], "valid_R20")
            ats = get_average_score(test_list[: ti + 1], "test_R20")

            # 根据配置选择验证集或测试集的平均分数
            if a.eval_average_acc:
                vs = avs
                ts = ats
            else:
                vs = vl[ti]["valid_R20"]
                ts = test_list[ti]["test_R20"]

            # 获取新用户的评估结果
            print("\t[The Result of new users in " + str(ti) + "-th Block]")
            nur = get_eval_with_mat(nu_mt, nu_vm, nu_tm, Tsmat, a.k_list)
            nurt = (
                "valid_R20 = "
                + str(nur["valid"]["R20"])
                + ", test_R20 = "
                + str(nur["test"]["R20"])
            )
            print("\t" + nurt + "\n")

            # 如果当前得分高于历史最佳得分，更新最佳模型
            if vs > ea["best_score"]:
                print(
                    "\t[Best Model Changed]\n\tvalid_score = "
                    + str(round(vs, 4))
                    + ", test_score = "
                    + str(round(ts, 4))
                )
                ea["best_score"] = vs
                ea["test_score"] = ts
                ea["avg_valid_score"] = avs
                ea["avg_test_score"] = ats

                ea["best_epoch"] = e
                ea["best_model"] = deepcopy(
                    {k: v.cpu() for k, v in T.state_dict().items()}
                )
                ea["score_mat"] = deepcopy(Tsm)
                ea["sorted_mat"] = deepcopy(Tsmat)
                ea["patience"] = 0

                bnur = deepcopy(nurt)

            # patience 机制，用于早停，不知道哪里来的，文中没有提到
            else:
                ea["patience"] += 1
                if ea["patience"] >= a.early_stop:
                    print("[Early Stopping]")
                    break

            # 如果使用了回放学习，更新回放学习数据集
            if a.replay_learning and e > 0:
                replay_learning_dataset = get_total_replay_learning_dataset_Teacher(
                    Tsm, Ssm, Srm, Psm, Prm, CLsm, CLrm, a
                )
                if mt == "VAE":
                    tl = get_VAE_replay_learning_loader_integrate_with_R(
                        replay_learning_dataset, pR, ptu, mi, a
                    )

        gc.collect()
        torch.cuda.empty_cache()

    # 打印最终结果
    print("\n[Final result of teacher's update in the " + str(ti) + "-th data block]")
    print(
        "best_epoch = "
        + str(ea["best_epoch"])
        + ", valid_score = "
        + str(ea["best_score"])
        + ", test_score = "
        + str(ea["test_score"])
    )
    print(
        "avg_valid_score = "
        + str(ea["avg_valid_score"])
        + ", avg_test_score = "
        + str(ea["avg_test_score"])
    )

    get_CL_result(
        ttd,
        tvd,
        ttds,
        ea["sorted_mat"],
        a.k_list,
        current_task_idx=a.num_task,
        FB_flag=False,
        return_value=False,
    )

    print("\t[The Result of new users in " + str(ti) + "-th Block]")
    print("\t" + bnur + "\n")
    ea["base_model"] = {
        basemodel(k): v.cpu()
        for k, v in ea["best_model"].items()
        if k.startswith("base_model")
    }

    return ea


def basemodel(x):
    return x[11:]


def main(args):
    """Main function for training and evaluation in Stage3:Teacher update"""

    gpu = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    scaler = torch.cuda.amp.GradScaler()
    # ipdb.set_trace()
    # Validate dataset and model
    assert args.dataset in [
        "Gowalla",
        "Yelp",
    ], "Dataset must be either 'Gowalla' or 'Yelp'."

    model_type = args.teacher
    model_seed = 0

    # Random Seed
    print(f"Random_seed = {model_seed}")
    set_random_seed(int(model_seed))

    ## Load data
    data_path = f"dataset/{args.dataset}/total_blocks_timestamp.pickle"
    data_dict_path = f"dataset/{args.dataset}"

    total_blocks = load_pickle(data_path)
    max_item = load_pickle(data_path)[-1].item.max() + 1
    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = (
        load_data_as_dict(data_dict_path, num_task=args.num_task)
    )

    task_idx = args.target_task
    distillation_idx = task_idx - 1

    p_block = total_blocks[task_idx]
    b_block = total_blocks[task_idx - 1]

    p_total_user = p_block.user.max() + 1
    p_total_item = p_block.item.max() + 1
    b_total_user = b_block.user.max() + 1
    b_total_item = b_block.item.max() + 1

    num_new_user = p_total_user - b_total_user
    num_new_item = p_total_item - b_total_item

    b_train_dict = total_train_dataset[
        f"TASK_{task_idx - 1}"
    ]  # {u_1 : {i_1, i_2, i_3}, ..., }
    _, b_train_mat, _, _ = get_train_valid_test_mat(
        task_idx - 1, total_train_dataset, total_valid_dataset, total_test_dataset
    )
    p_train_interaction, p_train_mat, p_valid_mat, p_test_mat = (
        get_train_valid_test_mat(
            task_idx, total_train_dataset, total_valid_dataset, total_test_dataset
        )
    )

    b_R = make_R(b_total_user, b_total_item, b_train_mat)
    p_R = make_R(p_total_user, p_total_item, p_train_mat)

    p_user_ids = torch.tensor(sorted(p_block.user.unique()))
    p_item_ids = torch.tensor(sorted(p_block.item.unique()))
    b_user_ids = torch.tensor(sorted(list(b_train_dict.keys())))  # .to(gpu)
    b_item_ids = torch.tensor(
        sorted(total_item_list[f"TASK_{task_idx - 1}"])
    )  # .to(gpu)

    _, b_user_mapping, b_item_mapping, b_rating_mat, UU, II = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    # Train/test/valid data split for new users/items
    new_user_train_mat, new_user_valid_mat, new_user_test_mat = (
        get_train_valid_test_mat_for_new_users(
            b_total_user, p_total_user, p_train_mat, p_valid_mat, p_test_mat
        )
    )

    ## Load Models
    if args.S_load_path is None:
        args.S_load_path = f"ckpts/{args.dataset}/students/{args.student}/Test"

    if args.T_load_path is None:
        args.T_load_path = f"ckpts/{args.dataset}/teachers/{model_type}"  # The student should be specified because the student and teacher collaboratively evolve along the data stream in our proposed CCD framework.

    print(
        f"Student = {args.student} (with low dimensionailty), S_load_path = {args.S_load_path}"
    )
    print(
        f"Teacher = {args.teacher} (with high dimensionailty), T_load_path = {args.T_load_path}"
    )

    # Load the path of student-side models (S_proxy, P_proxy, Student)
    load_D_model_dir_path = f"{args.S_load_path}/Distilled"
    load_S_model_dir_path = f"{args.S_load_path}/Stability"
    load_P_model_dir_path = f"{args.S_load_path}/Plasticity"
    load_CL_model_dir_path = f"{args.S_load_path}/CL"

    # Load the path of teacher
    RRD_SM_dir_path = f"{args.T_load_path}/"

    # Load teacher
    # RRD_SM_path = f"{RRD_SM_dir_path}/TASK_{distillation_idx}_score_mat.pth"
    # if distillation_idx == 0:
    RRD_SM_path = (
        f"ckpts/{args.dataset}/teachers/{model_type}/TASK_{args.target_task-1}.pth"
    )
    T_score_mat = (
        torch.load(RRD_SM_path, map_location=gpu)["score_mat"].detach().cpu()
    )  # RRD_SM[f"TASK_{distillation_idx}"]
    FT_score_mat = filtering_simple(T_score_mat, b_train_dict).detach().cpu()
    T_RRD_interesting_items = torch.topk(FT_score_mat, k=40, dim=1).indices
    negatvie_exclude = T_RRD_interesting_items.clone().detach()
    del RRD_SM_path, T_score_mat, FT_score_mat, T_RRD_interesting_items

    # Load students
    S_score_mat, P_score_mat, CL_score_mat = None, None, None

    # S proxy
    if args.Using_S:
        S_model_task_path = os.path.join(
            load_S_model_dir_path, f"TASK_{distillation_idx}.pth"
        )
        # ipdb.set_trace()
        _, S_score_mat, S_sorted_mat = load_saved_model(S_model_task_path, gpu)

        print("\n[Evaluation for S_proxy]")
        get_CL_result(
            total_train_dataset,
            total_valid_dataset,
            total_test_dataset,
            S_sorted_mat,
            args.k_list,
            current_task_idx=task_idx,
            FB_flag=False,
            return_value=False,
        )

        negatvie_exclude = torch.cat(
            [negatvie_exclude, torch.tensor(S_sorted_mat[:, :40])], dim=1
        )
        del S_sorted_mat

    # P proxy
    if (distillation_idx > 0 and args.Using_P) or (
        distillation_idx == 0 and args.Using_P and args.Using_S != True
    ):  # and args.P_model_path != "None":
        P_model_task_path = os.path.join(
            load_P_model_dir_path, f"TASK_{distillation_idx}.pth"
        )
        _, P_score_mat, P_sorted_mat = load_saved_model(P_model_task_path, gpu)

        print("\n[Evaluation for P_proxy]")
        get_CL_result(
            total_train_dataset,
            total_valid_dataset,
            total_test_dataset,
            P_sorted_mat,
            args.k_list,
            current_task_idx=task_idx,
            FB_flag=False,
            return_value=False,
        )

        negatvie_exclude = torch.cat(
            [negatvie_exclude, torch.tensor(P_sorted_mat[:, :40])], dim=1
        )
        del P_sorted_mat

        if distillation_idx == 0 and args.Using_P and args.Using_S != True:
            args.P_sample = args.S_sample

    # Student via continual update
    if args.Using_CL:
        CL_model_task_path = os.path.join(
            load_CL_model_dir_path, f"TASK_{task_idx}.pth"
        )
        _, CL_score_mat, CL_sorted_mat = load_saved_model(CL_model_task_path, gpu)

        print("\n[Evaluation for CL_Student]")
        get_CL_result(
            total_train_dataset,
            total_valid_dataset,
            total_test_dataset,
            CL_sorted_mat,
            args.k_list,
            current_task_idx=task_idx,
            FB_flag=False,
            return_value=False,
        )

        u_size, i_size = negatvie_exclude.shape
        negatvie_exclude_expand = torch.full((p_total_user, i_size), -1.0)
        negatvie_exclude_expand[:u_size, :i_size] = negatvie_exclude
        # ipdb.set_trace()
        negatvie_exclude = torch.cat(
            [negatvie_exclude_expand, torch.tensor(CL_sorted_mat[:, :40])], dim=1
        )
        del CL_sorted_mat

    # Dataset / DataLoader
    if model_type in ["TransformerSelf"]:
        train_dataset = implicit_CF_dataset(
            p_total_user,
            p_total_item,
            p_train_mat,
            args.nns,
            p_train_interaction,
            negatvie_exclude,
        )
    else:
        train_dataset = implicit_CF_dataset_AE(
            p_total_user, max_item, p_train_mat, is_user_side=True
        )
    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, drop_last=False
    )

    # Load Teacher model
    Teacher, T_sorted_mat = get_teacher_model(
        model_type, b_total_user, b_total_item, b_R, task_idx, max_item, gpu, args
    )
    print(f"\n[Teacher]\n{Teacher}")

    print("\n[[Before Update] Evalutation for Teacher]")
    get_CL_result(
        total_train_dataset,
        total_valid_dataset,
        total_test_dataset,
        T_sorted_mat,
        args.k_list,
        current_task_idx=task_idx,
        FB_flag=False,
        return_value=False,
    )

    # Increase the model size due to new users/items.
    Teacher, T_score_mat, T_sorted_mat = Teacher_update(
        model_type,
        Teacher,
        b_total_user,
        b_total_item,
        b_user_ids,
        b_item_ids,
        b_train_dict,
        p_total_user,
        p_total_item,
        p_R,
        p_user_ids,
        p_item_ids,
        num_new_user,
        num_new_item,
        gpu,
        train_loader,
        args,
    )

    print("\n[[After Update] Evalutation for Teacher]")
    get_CL_result(
        total_train_dataset,
        total_valid_dataset,
        total_test_dataset,
        T_sorted_mat,
        args.k_list,
        current_task_idx=task_idx,
        FB_flag=False,
        return_value=False,
    )

    ################################### Compose the initial dataset of replay learning (it adaptively changes through training) ##########################################################################################################################################################

    if args.replay_learning:
        if model_type in ["TransformerSelf"]:
            S_score_mat = (
                torch.sigmoid(S_score_mat) if S_score_mat is not None else None
            )
            P_score_mat = (
                torch.sigmoid(P_score_mat) if P_score_mat is not None else None
            )
            CL_score_mat = (
                torch.sigmoid(CL_score_mat) if CL_score_mat is not None else None
            )

        else:
            S_score_mat = (
                F.softmax(S_score_mat, dim=-1) if S_score_mat is not None else None
            )
            P_score_mat = (
                F.softmax(P_score_mat, dim=-1) if P_score_mat is not None else None
            )
            CL_score_mat = (
                F.softmax(CL_score_mat, dim=-1) if CL_score_mat is not None else None
            )

        S_rank_mat = (
            convert_to_rank_mat(S_score_mat) if S_score_mat is not None else None
        )
        P_rank_mat = (
            convert_to_rank_mat(P_score_mat) if P_score_mat is not None else None
        )
        CL_rank_mat = (
            convert_to_rank_mat(CL_score_mat) if CL_score_mat is not None else None
        )

        replay_learning_dataset = get_total_replay_learning_dataset_Teacher(
            T_score_mat,
            S_score_mat,
            S_rank_mat,
            P_score_mat,
            P_rank_mat,
            CL_score_mat,
            CL_rank_mat,
            args,
        )

        # If the model is VAE, we use pseudo-labeling by imputing the replay_learning_dataset with args.VAE_replay_learning_value.
        if model_type in ["VAE"]:
            train_loader = get_VAE_replay_learning_loader_integrate_with_R(
                replay_learning_dataset, p_R, p_total_user, max_item, args
            )

    eval_args = teacher_update(
        T=Teacher,
        mt=model_type,
        tl=train_loader,
        sc=scaler,
        a=args,
        ttd=total_train_dataset,
        tvd=total_valid_dataset,
        ttds=total_test_dataset,
        Tsm=T_score_mat,
        Ssm=S_score_mat,
        Srm=S_rank_mat,
        Psm=P_score_mat,
        Prm=P_rank_mat,
        CLsm=CL_score_mat,
        CLrm=CL_rank_mat,
        pR=p_R,
        ptu=p_total_user,
        mi=max_item,
        ti=task_idx,
        nu_mt=new_user_train_mat,
        nu_vm=new_user_valid_mat,
        nu_tm=new_user_test_mat,
        g=gpu,
    )

    # Model save
    if args.save:
        print("\n[Model Save]")

        if args.save_path is not None:
            save_path = args.save_path
        else:
            save_path = args.T_load_path

        Teacher_dir_path = save_path  # model name

        if not os.path.exists(Teacher_dir_path):
            os.makedirs(Teacher_dir_path)
        save_model(
            Teacher_dir_path,
            args.target_task,
            {
                "best_model": eval_args["best_model"],
                "score_mat": eval_args["score_mat"],
                "checkpoint": eval_args["base_model"],
            },
        )

        print("Teacher_dir_path", Teacher_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument(
        "--dataset", "--d", type=str, default=None, help="Gowalla or Yelp"
    )

    # model path
    parser.add_argument(
        "--save",
        "--s",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--s or --no-s)",
    )
    parser.add_argument(
        "--save_path",
        "--sp",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--S_model_path", type=str, default="ckpts/New_Student/Stability"
    )
    parser.add_argument(
        "--P_model_path", type=str, default="ckpts/New_Student/Plasticity"
    )
    parser.add_argument("--CL_model_path", type=str, default="ckpts/New_Student/CL")
    parser.add_argument("--RRD_SM_path", type=str, default="ckpts/New_Teacher/Ensemble")
    parser.add_argument(
        "--tcp",
        type=str,
        default="ckpts/New_Teacher",
        help="Teacher Ckpt Path",
    )

    # etc
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k_list", type=list, default=[20, 50, 100])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=0.0001)
    parser.add_argument("--bs", help="batch_size", type=int, default=2048)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--rs", type=int, default=1, help="random seed")
    parser.add_argument("--early_stop", type=int, default=2)
    parser.add_argument(
        "--nns", help="the number of negative sample", type=int, default=1
    )
    parser.add_argument("--td", help="teacher embedding dims", type=int, default=64)

    # LWCKD + PIW
    parser.add_argument("--nc", help="num_cluster", type=int, default=10)
    parser.add_argument("--T", help="temperature", type=int, default=5.0)
    parser.add_argument("--LWCKD_lambda", type=float, default=0.01)
    parser.add_argument("--cluster_lambda", type=float, default=1)

    # Method
    parser.add_argument("--eps", type=float, default=0.0001)
    parser.add_argument(
        "--annealing", "--a", action="store_true", default=True, help="using annealing"
    )
    parser.add_argument(
        "--random_init",
        "--r",
        action="store_true",
        default=False,
        help="random_initalization for new user/items",
    )
    parser.add_argument("--absolute", type=float, default=100)
    parser.add_argument("--replay_learning_lambda", type=float, default=0.5)

    # initalization for new users/items
    parser.add_argument("--init_topk", type=int, default=20)
    parser.add_argument(
        "--only_one_hop",
        "--ooh",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--ooh or --no-ooh)",
    )

    # eval
    parser.add_argument("--num_task", type=int, default=6)
    parser.add_argument(
        "--eval_average_acc",
        "--eaa",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--eaa or --no-eaa)",
    )
    parser.add_argument(
        "--hyper_param",
        "--h",
        action=argparse.BooleanOptionalAction,
        help="whether saving param or not (--h or --no-h)",
    )

    # LightGCN
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument(
        "--using_layer_index", type=str, default="avg", help="first, last, avg"
    )
    parser.add_argument("--VAE_replay_learning_value", type=float, default=0.1)
    parser.add_argument("--kl_lambda", "-kl", type=float, default=1.0)

    # S & P proxies
    parser.add_argument(
        "--target_task", "--tt", help="target_task", type=int, default=-1
    )
    parser.add_argument(
        "--replay_learning",
        "--rl",
        action=argparse.BooleanOptionalAction,
        help="whether using param or not (--replay_learning or --no-replay_learning)",
    )
    parser.add_argument(
        "--Using_S",
        "--US",
        action=argparse.BooleanOptionalAction,
        help="whether using stablity proxy or not (--US or --no-US)",
    )
    parser.add_argument(
        "--Using_P",
        "--UP",
        action=argparse.BooleanOptionalAction,
        help="whether using plasticity proxy or not (--UP or --no-UP)",
    )
    parser.add_argument(
        "--Using_CL",
        "--UCL",
        action=argparse.BooleanOptionalAction,
        help="whether using student or not (--UCL or --no-UCL)",
    )
    parser.add_argument(
        "--S_sample", "--ss", type=int, default=5, help="# for stability proxy"
    )
    parser.add_argument(
        "--P_sample", "--ps", type=int, default=5, help="# for plasticity proxy"
    )
    parser.add_argument(
        "--CL_sample", "--cs", type=int, default=5, help="# for student"
    )
    parser.add_argument("--S_load_path", "--Slp", type=str, default=None)
    parser.add_argument("--T_load_path", "--Tlp", type=str, default=None)
    parser.add_argument(
        "--student",
        type=str,
        default=None,
        help="TransformerSelf, GCNSelf, MFSelf or VAESelf",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="TransformerSelf, GCNSelf, MFSelf or VAESelf",
    )

    args = parser.parse_args()

    # Teacher's embedding dimension
    # if args.dataset == "Gowalla":
    #     args.td = 64
    # elif args.dataset == "Yelp":
    #     args.td = 128

    print_command_args(args)
    main(args)
    print_command_args(args)

    # run code: python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_0 --tt 5 --rl --UCL --US --UP --ab 100 --ss 1 --ps 5 --cs 1 --max_epoch 10
