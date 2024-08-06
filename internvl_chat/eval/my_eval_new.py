import sys
import json
import os
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import argparse

def load_db(infile, is_person_flag):
    q_dict = {}
    sparse_dis = []
    semantic_dis = []
    sparse_mean, sparse_std = 0.20875011283749365, 0.1387529783792963 
    semantic_mean, semantic_std = 0.5025411223127866, 0.13310462257703584

    # 读取 JSONL 文件
    with open(infile, 'r') as f:
        for line in f:
            try:
                row = json.loads(line)
                sid = row['searchid']
                label = row['label']
                sparse_ctr = row['sparse_ctr']
                semantic_ctr = row['semantic_ctr']
                label = int(label)
                try: 
                    sparse_ctr = float(sparse_ctr) 
                    semantic_ctr = float(semantic_ctr)
                except:
                    sparse_ctr = float(row.get('response', -1))
                    semantic_ctr = float(row.get('response', -1))
                prefeedid_key = row['pre_feedid']
                is_person = row['is_person']
                predict_label = row.get('response', -1)
                try:
                    predict_label = int(predict_label)
                except:
                    predict_label = -1
            except:
                continue
            if sid == "None" or sid == "":
                continue

            if int(is_person) == is_person_flag:
                continue

            sid = str(sid) + '_' + str(prefeedid_key) + '_' + str(is_person)

            if sid not in q_dict:
                q_dict[sid] = []
            q_dict[sid].append([label, sparse_ctr, semantic_ctr, prefeedid_key, predict_label])
            sparse_dis.append(sparse_ctr)
            semantic_dis.append(semantic_ctr)
    return q_dict, sparse_dis, semantic_dis

def get_pair_acc(item, eps=0):
    item.sort(reverse=True)
    item_size = len(item)
    pair_cnt = 0
    sparse_right = 0
    semantic_right = 0

    for i in range(item_size):
        label_i = item[i][0]
        sparse_ctr_i = item[i][1]
        semantic_ctr_i = item[i][2]
        for j in range(i+1, item_size):
            label_j = item[j][0]
            sparse_ctr_j = item[j][1]
            semantic_ctr_j = item[j][2]
            if label_i > label_j:
                pair_cnt += 1
                if sparse_ctr_i - sparse_ctr_j > eps:
                    sparse_right += 1
                if semantic_ctr_i - semantic_ctr_j > eps:
                    semantic_right += 1
    return pair_cnt, sparse_right, semantic_right

def get_acc(item):
    item_size = len(item)
    cnt = 0
    right = 0
    for i in range(item_size):
        label_i = item[i][0]
        p_label = item[i][4]
        cnt += 1
        if label_i == p_label:
            right += 1
    return cnt, right
    
def get_auc(item, fun=None):
    y_true = list(map(lambda x:fun(x[0]) if fun else x[0], item))
    y_sparse_score = list(map(lambda x:x[1], item))
    y_semantic_score = list(map(lambda x:x[2], item))
    sparse_auc = roc_auc_score(y_true, y_sparse_score)
    semantic_auc = roc_auc_score(y_true, y_semantic_score)
    return sparse_auc, semantic_auc

def process(infile, is_person_flag, eps_flag, fun=None):
    q_dict, sparse_dis, semantic_dis = load_db(infile, is_person_flag)
    q_size = len(q_dict)
    all_pair_cnt = 0
    all_sparse_right = 0
    all_semantic_right = 0
    all_sparse_auc = 0
    all_semantic_auc = 0
    auc_cnt = 0
    q_cnt = 0
    q_acc = 0
    cnt = 0
    right = 0
    
    acc_cnt = 0
    acc_right = 0
    for q in q_dict:
        q_pair_cnt, q_sparse_right, q_semantic_right = get_pair_acc(q_dict[q], eps=eps_flag)
        tmp_cnt, tmp_right = get_acc(q_dict[q])
        cnt += tmp_cnt
        right += tmp_right
        
        if q_pair_cnt > 0:
            tmp_acc = q_semantic_right * 1.0 / q_pair_cnt
            q_cnt += 1
            q_acc += tmp_acc
        all_pair_cnt += q_pair_cnt
        all_sparse_right += q_sparse_right
        all_semantic_right += q_semantic_right
        try:
            sparse_auc, semantic_auc = get_auc(q_dict[q], fun)
            if sparse_auc >= 0 and sparse_auc <= 1.0 and semantic_auc >= 0 and semantic_auc <= 1.0:
                all_sparse_auc += sparse_auc
                all_semantic_auc += semantic_auc
                auc_cnt += 1
        except:
            continue

    return f'{right * 1.0 / cnt:.2f}', f'{all_sparse_right * 1.0 / all_pair_cnt:.2f}', f'{q_acc * 1.0 / q_cnt:.2f}', f'{all_sparse_auc / auc_cnt:.2f}'

def refine_label(label):
    if label > 0:
        label = 1
    else:
        label = 0
    return label 

def refine_no_label(label):
    if label >= 2:
        label = 1
    else:
        label = 0
    return label

parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument('--ass_file', type=str, required=True, default="/Users/tangbinghao/Evaluation/minicpm-v-2/minicpm-v-2-epoch5.jsonl", help='Path to the assessment file')
args = parser.parse_args()
ass_file = args.ass_file

is_person_flags = [1, 0]
eps_flags = [0.0, 0.1, 0.2, 0.3, 0.4]

print(f"{'is_person':>10} {'eps':>5} {'single_acc':>12} {'pair_acc':>10} {'group_acc':>11} {'group_auc':>10}")
for is_person in is_person_flags:
    for eps in eps_flags:
        singel_acc, pair_acc, group_acc, group_auc = process(ass_file, is_person, eps)
        print(f"{(1-is_person):>10} {eps:>5} {singel_acc:>12} {pair_acc:>10} {group_acc:>11} {group_auc:>10}")

# 计算准确率
fr = open(ass_file, 'r')
predictions_person = []
labels_person = []
predictions_non_person = []
labels_non_person = []
for data in fr:
    data = data.strip()
    data = json.loads(data)
    if int(data['is_person']) == 1:
        predictions_person.append(data['response'].replace(' ',''))
        labels_person.append(data['label'])
    else:
        predictions_non_person.append(data['response'].replace(' ',''))
        labels_non_person.append(data['label'])

acc_person = accuracy_score(labels_person, predictions_person)
acc_non_person = accuracy_score(labels_non_person, predictions_non_person)
print(f"Accuracy for is_person=1: {acc_person}")
print(f"Accuracy for is_person=0: {acc_non_person}")