import csv
import json
import sys
import os
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import math
import random


# k-core过滤
def k_core(data, k=10):
    """
    Filter out the users and items that have less than k interactions.
    """
    filtered_data = data.copy()
    while True:
        # 统计每个用户的检查次数
        user_counts = filtered_data['user'].value_counts()
        # 统计每个位置的检查次数
        location_counts = filtered_data['item'].value_counts()
        # 过滤活跃用户和位置
        active_users = user_counts[user_counts >= k].index
        active_locations = location_counts[location_counts >= k].index
        # 过滤数据
        new_filtered_data = filtered_data[
            filtered_data['user'].isin(active_users) &
            filtered_data['item'].isin(active_locations)
            ]
        # 检查是否有变化
        if len(new_filtered_data) == len(filtered_data):
            break  # 如果没有变化，则退出循环
        filtered_data = new_filtered_data  # 更新过滤后的数据
    # 查看最终过滤后的数据集
    return filtered_data


    user_count = data['user'].value_counts()
    item_count = data['item'].value_counts()
    data = data[data['user'].isin(user_count[user_count >= k].index)]
    data = data[data['item'].isin(item_count[item_count >= k].index)]
    return data

# 划分数据集
def train_valid_test_split(df_task, valid_size=0.1, test_size=0.1):
    train_dict, test_dict, valid_dict = {}, {}, {}
    # 根据比例初步划分
    for user_id, group in df_task.groupby('user'):
        items = list(group['item'].unique())
        np.random.shuffle(items)

        n_valid = math.ceil(len(items) * valid_size)
        n_test = math.ceil(len(items) * test_size)

        valid_list = items[:n_valid]
        test_list = items[n_valid:n_valid + n_test]
        train_list = items[n_valid + n_test:]
        # assign
        train_dict[user_id] = train_list
        valid_dict[user_id] = valid_list
        test_dict[user_id] = test_list

    # 过滤掉valid和test中没有出现在train中的item
    train_mat_R = defaultdict(list)
    for user in train_dict:
        for item in train_dict[user]:
            train_mat_R[item].append(user)

    for u in list(valid_dict.keys()):
        for i in list(valid_dict[u]):
            if i not in train_mat_R:
                valid_dict[u].remove(i)

        if len(valid_dict[u]) == 0:
            del valid_dict[u]
            del test_dict[u]

    for u in list(test_dict.keys()):
        for i in list(test_dict[u]):
            if i not in train_mat_R:
                test_dict[u].remove(i)

        if len(test_dict[u]) == 0:
            del valid_dict[u]
            del test_dict[u]

    item_list = list(train_mat_R.keys())
    num_base_block_users=max(train_dict.keys())+1
    num_base_block_items = max(item_list) + 1


    return {"train_dict": defaultdict(list, train_dict),
            "valid_dict": defaultdict(list, valid_dict),
            "test_dict": defaultdict(list, test_dict),
            "num_base_block_users":num_base_block_users,
            "num_base_block_items":num_base_block_items,
            "item_list": item_list}

def save_pickle(data, file_path):
    directory = os.path.dirname(file_path)
    print("create pickle",file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':

    data_block_path= f"../dataset/Gowalla_new0/total_blocks_timestamp.pickle"
    data_dict_path= f"../dataset/Gowalla_new0/"

    # 读取数据
    df_gowalla=pd.read_csv("../loc-gowalla_totalCheckins.txt/gowalla_totalCheckins.txt",sep='\t',header=None)
    df_gowalla.columns=['user','timestamp','latitude','longitude','item']
    df_gowalla=df_gowalla[['user','item','timestamp']]

    # 过滤重复的用户和位置交互
    df_gowalla = df_gowalla.drop_duplicates(subset=['user', 'item'])
    # 过滤交互次数少于10的用户和位置
    filter_gowalla = k_core(df_gowalla, k=10)
    print(len(filter_gowalla))# 和table1一致
    # 按时间排序
    filter_gowalla=filter_gowalla.sort_values(by=['timestamp'])
    # 重新编号
    filter_gowalla['user'], _ = pd.factorize(filter_gowalla['user'])
    filter_gowalla['item'], _ = pd.factorize(filter_gowalla['item'])
    # 划分block .根据table1分的，，很奇怪，，，
    task_0 = filter_gowalla.iloc[:int(len(filter_gowalla) * 0.5)]
    task_1 = filter_gowalla.iloc[int(len(filter_gowalla) * 0.5):576068]
    task_2 = filter_gowalla.iloc[576068:688235]
    task_3 = filter_gowalla.iloc[688235:811277]
    task_4 = filter_gowalla.iloc[811277:937751]
    task_5 = filter_gowalla.iloc[937751:]

    Task=[task_0,task_1,task_2,task_3,task_4,task_5]
    total_data = []

    for task_idx in range(len(Task)):
        total_data.append(Task[task_idx])# ->6个dataframe
        task_data_dict_path = os.path.join(data_dict_path, f"TASK_{task_idx}.pickle")
        task_data = train_valid_test_split(Task[task_idx])
        save_pickle(task_data, task_data_dict_path)

    save_pickle(total_data, data_block_path)


