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
def teacher_update(T, mt, tl, sc, a, ttd,
                    tvd, ttds, Tsm, Ssm, Srm, Psm,
                    Prm, CLsm, CLrm, pR, ptu, mi, ti,
                    nu_mt, nu_vm, nu_tm, g):
    ## 1.z

    # 根据模型类型配置优化器参数和loss类型
    if mt == "BPR" or mt == "LightGCN":
        # 对于BPR或LightGCN，将模型参数和聚类参数包装在一起
        p = [{"params": T.parameters()}, {"params": T.cluster}]
        lt = ["base", "UI", "IU", "UU", "II", "cluster"]
    else:
        if mt == "VAE":
            # 对于VAE，只需要模型的主要参数
            p = T.parameters()
            lt = ["base", "kl"]

    # 初始化Adam优化器
    opt = optim.Adam(p, lr=a.lr, weight_decay=a.reg)
    # 使用二元交叉熵loss函数
    crit = nn.BCEWithLogitsLoss(reduction='sum')
    # 初始化评估参数
    ea = {"best_score": 0, "test_score": 0, "best_epoch": 0, "best_model": None, "score_mat": None, "sorted_mat": None, "patience": 0, "avg_valid_score": 0, "avg_test_score": 0}
    tt = 0
    
    gc.collect()
    torch.cuda.empty_cache()
    # 训练
    for e in range(a.max_epoch):
        print("\n[Epoch:" + str(e + 1) + "/" + str(a.max_epoch) + "]")
        # 初始化每种loss类型的累计值
        el = {}
        for l in lt:
            el["epoch_" + l + "_loss"] = 0.0
        erl = 0.0
        st = time.time()
        # 负采样
        tl.dataset.negative_sampling()

        T.train()
        for mb in tl:
            # 正向传播计算loss
            if mt == "BPR" or mt == "LightGCN":
                bl, ul, il, uul, iil, cl = T(mb)
                bl = bl + a.LWCKD_lambda * (ul + il + uul + iil) + (a.cluster_lambda * cl)
            else:
                if mt == "VAE":
                    bl, kl = T(mb)
                    bl = bl + a.kl_lambda * kl
            # 反向传播和优化
            opt.zero_grad()
            sc.scale(bl).backward()
            sc.step(opt)
            sc.update()
            # 更新loss
            for l in lt:
                el["epoch_" + l + "_loss"] += eval(l + "_loss").item()

        for l in lt:
            ln = "epoch_" + l + "_loss"
            el[ln] = round(el[ln] / len(tl), 4)

        cft = time.time()
        print(str(el) + ", CF_time = " + str(round(cft - st, 4)) + " seconds", end=" ")

        ##2. 回放学习部分
        if a.replay_learning and (mt == "BPR" or mt == "LightGCN"):
            # 判断是否使用退火技术来调整回放学习的权重
            if a.annealing:
                # 使用指数退火调整lambda，随着时间的推移逐渐减少回放学习的影响
                rll = a.replay_learning_lambda * torch.exp(torch.tensor(-e) / a.T)
            else:
                rll = a.replay_learning_lambda
            
            # 打乱回放学习数据集
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
                bl = torch.tensor(rl[s:en], dtype=torch.float16).to(g)

                # 获取用户和项目的embedding
                ue, ie = T.base_model.get_embedding()
                bue = ue[bu]
                bie = ie[bi]

                # 计算loss
                o = (bue * bie).sum(1)
                bl = crit(o, bl)
                bl *= rll

                # 反向传播和优化
                opt.zero_grad()
                sc.scale(bl).backward()
                sc.step(opt)
                sc.update()
                erl += bl.item()

            rlt = time.time()
            print("epoch_replay_learning_loss = " + str(round(erl / it, 4)) + ", replay_learning_lambda = " + str(round(rll.item(), 5)) + ", replay_learning_time = " + str(round(rlt - cft, 4)) + " seconds", end=" ")

        et = time.time()
        etime = et - st
        tt += etime
        print("epoch_time = " + str(round(etime, 4)) + " seconds, total_time = " + str(round(tt, 4)) + " seconds")

        ##3. 评估
        if e % 5 == 0:
            print("\n[Evaluation]")
            # 设置模型为评估模式
            T.eval()
            with torch.no_grad():
                # 计算得分矩阵
                if mt == "BPR" or mt == "LightGCN":
                    Tsm, Tsmat = get_sorted_score_mat(T, topk=1000, return_sorted_mat=True)
                else:
                    if mt == "VAE":
                        Tsm = get_score_mat_for_VAE(T.base_model, tl, g).detach().cpu()
                        Tsmat = to_np(torch.topk(Tsm, k=1000).indices)
            # 获取当前任务的评估结果
            vl, tl = get_CL_result(ttd, tvd, ttds, Tsmat, a.k_list, current_task_idx=ti, FB_flag=False, return_value=True)
            
            # 计算平均分数
            avs = get_average_score(vl[:ti + 1], "valid_R20")
            ats = get_average_score(tl[:ti + 1], "test_R20")

            # 根据配置选择验证集或测试集的平均分数
            if a.eval_average_acc:
                vs = avs
                ts = ats
            else:
                vs = vl[ti]["valid_R20"]
                ts = tl[ti]["test_R20"]

            # 获取新用户的评估结果
            print("\t[The Result of new users in " + str(ti) + "-th Block]")
            nur = get_eval_with_mat(nu_mt, nu_vm, nu_tm, Tsmat, a.k_list)
            nurt = "valid_R20 = " + str(nur["valid"]["R20"]) + ", test_R20 = " + str(nur["test"]["R20"])
            print("\t" + nurt + "\n")

            # 如果当前得分高于历史最佳得分，更新最佳模型
            if vs > ea["best_score"]:
                print("\t[Best Model Changed]\n\tvalid_score = " + str(round(vs, 4)) + ", test_score = " + str(round(ts, 4)))
                ea["best_score"] = vs
                ea["test_score"] = ts
                ea["avg_valid_score"] = avs
                ea["avg_test_score"] = ats

                ea["best_epoch"] = e
                ea["best_model"] = deepcopy({k: v.cpu() for k, v in T.state_dict().items()})
                ea["score_mat"] = deepcopy(Tsm)
                ea["sorted_mat"] = deepcopy(Tsmat)
                ea["patience"] = 0

                bnur = deepcopy(nurt)

            #patience 机制，用于早停，不知道哪里来的，文中没有提到
            else:
                ea["patience"] += 1
                if ea["patience"] >= a.early_stop:
                    print("[Early Stopping]")
                    break

            # 如果使用了回放学习，更新回放学习数据集
            if a.replay_learning and e > 0:
                replay_learning_dataset = get_total_replay_learning_dataset_Teacher(Tsm, Ssm, Srm, Psm, Prm, CLsm, CLrm, a)
                if mt == "VAE":
                    tl = get_VAE_replay_learning_loader_integrate_with_R(replay_learning_dataset, pR, ptu, mi, a)

        gc.collect()
        torch.cuda.empty_cache()

    # 打印最终结果
    print("\n[Final result of teacher's update in the " + str(ti) + "-th data block]")
    print("best_epoch = " + str(ea["best_epoch"]) + ", valid_score = " + str(ea["best_score"]) + ", test_score = " + str(ea["test_score"]))
    print("avg_valid_score = " + str(ea["avg_valid_score"]) + ", avg_test_score = " + str(ea["avg_test_score"]))

    get_CL_result(ttd, tvd, ttds, ea["sorted_mat"], a.k_list, current_task_idx=a.num_task, FB_flag=False, return_value=False)

    print("\t[The Result of new users in " + str(ti) + "-th Block]")
    print("\t" + bnur + "\n")

    return ea