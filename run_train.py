import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import SeqDataset
from models import BERT4Rec
from trainers import FinetuneTrainer
import wandb
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)

def random_neg(l, r, s):
    # log에 존재하는 아이템과 겹치지 않도록 sampling
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

if __name__ == "__main__":
    # model setting
    max_len = 50
    hidden_units = 50
    num_heads = 1
    num_layers = 2
    dropout_rate=0.5
    num_workers = 1
    device = 'cuda' 

    # training setting
    lr = 0.001
    batch_size = 128
    num_epochs = 100
    mask_prob = 0.15 # for cloze task

    ############# 중요 #############
    # data_path는 사용자의 디렉토리에 맞게 설정해야 합니다.
    data_path = '/opt/ml/input/data/train/train_ratings.csv'
    df = pd.read_csv(data_path)

    item_ids = df['item'].unique()
    user_ids = df['user'].unique()
    num_item, num_user = len(item_ids), len(user_ids)
    num_batch = num_user // batch_size

    # user, item indexing
    item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

    # dataframe indexing
    df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    df.sort_values(['user_idx', 'time'], inplace=True)
    del df['item'], df['user'] 

    # train set, valid set 생성
    users = defaultdict(list) # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
    user_train = {}
    user_valid = {}
    for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        users[u].append(i)

    for user in users:
        user_train[user] = users[user][:-1]
        user_valid[user] = [users[user][-1]]

    print(f'num users: {num_user}, num items: {num_item}')

    model = BERT4Rec(num_user, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate, device)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
    seq_dataset = SeqDataset(user_train, num_user, num_item, max_len, mask_prob)
    data_loader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True) # TODO4: pytorch의 DataLoader와 seq_dataset을 사용하여 학습 파이프라인을 구현해보세요.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, num_epochs + 1):
    tbar = tqdm(data_loader)
    for step, (log_seqs, labels) in enumerate(tbar):
        logits = model(log_seqs)
        
        # size matching
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1).to(device)
        
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        tbar.set_description(f'Epoch: {epoch:3d}| Step: {step:3d}| Train loss: {loss:.5f}')

    model.eval()

    NDCG = 0.0 # NDCG@10
    HIT = 0.0 # HIT@10

    num_item_sample = 100
    num_user_sample = 1000
    users = np.random.randint(0, num_user, num_user_sample) # 1000개만 sampling 하여 evaluation
    for u in users:
        seq = (user_train[u] + [num_item + 1])[-max_len:] # TODO5: 다음 아이템을 예측하기 위한 input token을 추가해주세요.
        rated = set(user_train[u] + user_valid[u])
        item_idx = [user_valid[u][0]] + [random_neg(1, num_item + 1, rated) for _ in range(num_item_sample)]

        with torch.no_grad():
            predictions = - model(np.array([seq]))
            predictions = predictions[0][-1][item_idx] # sampling
            rank = predictions.argsort().argsort()[0].item()
        
        if rank < 10: # @10
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1
    print(f'NDCG@10: {NDCG/num_user_sample}| HIT@10: {HIT/num_user_sample}')