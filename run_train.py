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
    print("start!")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--mask_prob', type=float, default=0.15)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--hidden_units', type=int, default=50)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_workers', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()                     
    
    # model setting
    device = 'cuda' 

    # training setting
    lr = 0.001
    #mask_prob = 0.2 # for cloze task

    ############# 중요 #############
    # data_path는 사용자의 디렉토리에 맞게 설정해야 합니다.
    output_dir = '/opt/ml/input/code/output/'
    data_path = '/opt/ml/input/data/train/train_ratings.csv'
    df = pd.read_csv(data_path)

    item_ids = df['item'].unique()
    user_ids = df['user'].unique()
    num_item, num_user = len(item_ids), len(user_ids)
    num_batch = num_user // args.batch_size

    # user, item indexing
    item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

    # dataframe indexing
    df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    df.sort_values(['user_idx', 'time'], inplace=True) # 시간 순서대로 정렬
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

    model = BERT4Rec(num_user, num_item, args.hidden_units, args.num_heads, args.num_layers, args.max_len, args.dropout_rate, device)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
    seq_dataset = SeqDataset(user_train, num_user, num_item, args.max_len, args.mask_prob)
    data_loader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True) # TODO4: pytorch의 DataLoader와 seq_dataset을 사용하여 학습 파이프라인을 구현해보세요.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # wandb.login()
    # wandb.init(group="BERT4Rec numlayers", project="MovieLens", entity="recsys-06", name=f"BERT4Rec_{args.num_layers}")
    # wandb.config = args

    for epoch in range(1, args.epochs + 1):
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
    #users = np.random.randint(0, num_user, num_user_sample) # 1000개만 sampling 하여 evaluation
    users = list(range(num_user))
    for u in users:
        # 최근 item 50개만 가져옴
        seq = (user_train[u] + [num_item + 1])[-args.max_len:] # TODO5: 다음 아이템을 예측하기 위한 input token을 추가해주세요.
        # u == 0(user id가 11)일 때
        # user_train[u] : [0, 1, 2,...., 375]
        # user_valid[u] [376]
        rated = set(user_train[u] + user_valid[u])
        # log에 존재하는 아이템과 겹치지 않도록 랜덤으로 sampling
        item_idx = [user_valid[u][0]] + [random_neg(1, num_item + 1, rated) for _ in range(num_item_sample)]
        with torch.no_grad():
            predictions = - model(np.array([seq]))
            predictions = predictions[0][-1][item_idx] # sampling # 안 본 영화의 prediction
            rank = predictions.argsort().argsort()[0].item() 

        if rank < 10: # @10
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1

    # wandb.log({
    #     'NDCG@10': NDCG/num_user_sample,
    #     'HIT@10' : HIT/num_user_sample
    # })

    print(f'NDCG@10: {NDCG/num_user_sample}| HIT@10: {HIT/num_user_sample}')

    print("@@@@@ submission @@@@@")
    model.eval()

    submission_user = list()
    submission_item = list()
    num_item_sample = 100
    num_user_sample = 1000
    #users = np.random.randint(0, num_user, num_user_sample) # 1000개만 sampling 하여 evaluation
    users = list(range(num_user))
    for u in users:
        # 최근 item 50개만 가져옴
        # 다음 아이템을 예측하기 위한 input token을 추가해주세요.
        seq = (user_train[u] + [num_item + 1])
        if len(seq) > args.max_len : seq = seq[-args.max_len:]
        # user_train[u] : [0, 1, 2,...., 375]
        # user_valid[u] [376]
        rated = set(user_train[u] + user_valid[u])
        # log에 존재하는 아이템과 겹치지 않도록 랜덤으로 sampling
        #item_idx = [user_valid[u][0]] + [random_neg(1, num_item + 1, rated) for _ in range(num_item_sample)]
        item_idx = [idx for idx in range(1, num_item+1) if idx not in rated]
        with torch.no_grad():
            predictions = - model(np.array([seq]))
            predictions = predictions[0][-1][item_idx] # sampling # 안 본 영화의 prediction
            #rank = predictions.argsort().argsort()[0].item() 
            top_items = predictions.argsort()[:10]
            for i in top_items:
                submission_item.append(item2idx.keys()[item_idx[i]-1])
        submission_user+= [user2idx.keys()[u]]*10
        
        # if rank < 10: # @10
        #     NDCG += 1 / np.log2(rank + 2)
        #     HIT += 1

    submission_user = np.array(submission_user).reshape(-1, 1)
    submission_item = np.array(submission_item).reshape(-1, 1)
    submission = np.hstack((submission_user, submission_item))
    pd.DataFrame(submission, columns=['user','item']).to_csv(os.path.join(output_dir,f'BERT4Rec_batchsize_{args.batch_size}.csv'), index=False)
    print(os.path.join(output_dir,f'BERT4Rec_batchsize_{args.batch_size}.csv'))