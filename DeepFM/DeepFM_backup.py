import argparse
import os

from modules import Encoder, LayerNorm

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch.optim as optim
import bottleneck as bn

from trainers import FinetuneTrainer
import wandb
from utils import (
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
    recall_at_k,
)


class RatingDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self._device = torch.device("cuda:0")
        self.input_tensor = input_tensor.long().to(self._device)
        self.target_tensor = target_tensor.long().to(self._device)

    def __getitem__(self, index):
        return self.input_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.target_tensor.size(0)


class TestDataset(Dataset):
    def __init__(self, df):
        self._device = torch.device("cuda:0")
        self.df = torch.tensor(df.values).to(self._device)

    def __len__(self):
        return self.df.size(0)

    def __getitem__(self, idx):
        return self.df[idx]


class DeepFM(nn.Module):
    def __init__(self, args, input_dims):
        super(DeepFM, self).__init__()
        self.mlp_dims = args.mlp_dims
        self.drop_rate = args.dropout

        total_input_dim = int(sum(input_dims))  # n_user + n_movie + n_time + n_year + 2 * n_genre

        # Fm component의 constant bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = nn.Embedding(total_input_dim, 1)

        self.embedding = nn.Embedding(total_input_dim, args.embedding_dim)
        self.embedding_dim = len(input_dims) * args.embedding_dim

        mlp_layers = []
        for i, dim in enumerate(self.mlp_dims):
            if i == 0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
            else:
                mlp_layers.append(nn.Linear(self.mlp_dims[i - 1], dim))
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(self.drop_rate))
        mlp_layers.append(nn.Linear(self.mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x):
        # x : (batch_size, total_num_input)
        embed_x = self.embedding(x)

        fm_y = self.bias + torch.sum(self.fc(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y

    def mlp(self, x):
        embed_x = self.embedding(x)
        inputs = embed_x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, x):
        embed_x = self.embedding(x)
        # fm component
        fm_y = self.fm(x).squeeze(1)

        # deep component
        mlp_y = self.mlp(x).squeeze(1)

        y = torch.sigmoid(fm_y + mlp_y)
        return y


def WandB():
    wandb.init(
        # 필수
        project="MovieLens",  # project Name
        entity="recsys-06",  # Repository 느낌 변경 X
        name="DEEPFM_batch1024_onehot",  # -> str : ex) "모델_파라티머_파라미터_파라미터", 훈련 정보에 대해 알아보기 쉽게
        notes="this is test",  # -> str commit의 메시지 처럼 좀 더 상세한 설명 log
        group="DeepFM",
        # 추가 요소
        # tags -> str[] baseline, production등 태그 기능.
        # save_code -> bool 코드 저장할지 말지 default false
        # group -> str : 프로젝트내 그룹을 지정하여 개별 실행을 더 큰 실험으로 구성, k-fold교차, 다른 여러 테스트 세트에 대한 모델 훈련 및 평가 가능.

        #  more info
        # https://docs.wandb.ai/v/ko/library/init
    )


def main():
    WandB()
    parser = argparse.ArgumentParser()

    # data directory
    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="DeepFM", type=str)
    parser.add_argument("--embedding_dim", default=10, type=int)
    parser.add_argument(
        "--mlp_dims", type=list, default=[200, 200, 200], help="define mlp layers"
    )
    # 몇 층으로 구성 되었는 지,
    # network shape (constant/increasing/decreasing/diamond),
    # 층 당 뉴런의 개수를 정의
    parser.add_argument("--negative", default=1, type=int) # Negative instance 생성 개수
    parser.add_argument("--activation", default="relu", type=str)  # gelu relu
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="hidden dropout p"
    )

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()
    wandb.config.update(args)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    device = torch.device("cuda:0") # TODO cuda

    # process 1 데이터 파일 로드 및 구성
    raw_rating_df = pd.read_csv(args.data_dir + 'train_ratings.csv')
    submission_user = raw_rating_df['user'].unique().repeat(10)
    submission_user = np.array(submission_user).reshape(-1, 1)
    raw_rating_df['time'] = pd.to_datetime(raw_rating_df['time'], unit='s').dt.year  # time(시청 연도)
    max_time_df = raw_rating_df.groupby(['user'])['time'].max()
    max_time_df = pd.DataFrame(max_time_df)
    max_time_df = max_time_df.reset_index()

    raw_year_df = pd.read_csv("../../data/train/years.tsv", sep='\t')  # year 결측치 해결
    year_df = pd.DataFrame({'item': [6987, 3310, 7243, 8511, 32898, 7065, 119145, 6988],
                            'year': [1920, 1921, 1916, 1917, 1902, 1915, 2015, 1919]})
    raw_year_df = pd.concat([raw_year_df, year_df], axis=0, sort=False)
    raw_rating_df = pd.merge(raw_rating_df, raw_year_df, left_on='item', right_on='item', how='inner')  # year(개봉 연도)

    raw_genre_df = pd.read_csv("../../data/train/genres.tsv", sep='\t')
    genre_items = list(set((raw_genre_df.loc[:, 'item'])))
    genre_items.sort()
    genre_dict = {genre: i for i, genre in enumerate(set(raw_genre_df['genre']))}
    genre_item2id = {genre_items[i]: i for i in range(len(genre_items))}
    genre_id2item = {i: genre_items[i] for i in range(len(genre_items))}
    raw_genre_df['genre'] = raw_genre_df['genre'].map(lambda x: genre_dict[x])
    raw_genre_df['item'] = raw_genre_df['item'].map(lambda x: genre_item2id[x])
    n_item = len(genre_items)
    genre_onehot = torch.zeros((n_item, 18))
    for i in range(len(raw_genre_df)):
        item_id, genre_id = raw_genre_df.iloc[i]
        genre_onehot[item_id][genre_id] += 1
    genre_onehot = pd.DataFrame(genre_onehot)
    genre_onehot['item'] = genre_onehot.index
    genre_onehot['item'] = genre_onehot['item'].map(lambda x: genre_id2item[x])
    raw_rating_df = pd.merge(raw_rating_df, genre_onehot, left_on='item', right_on='item', how='left')

    raw_rating_df['rating'] = 1.0  # implicit feedback

    users = set(raw_rating_df.loc[:, 'user']) # user 집합
    items = set(raw_rating_df.loc[:, 'item']) # item 집합

    # 3. Negative instance 생성
    print("Create Negative instances")
    num_negative = args.negative
    user_group_dfs = list(raw_rating_df.groupby('user')['item'])
    first_row = True
    user_neg_dfs = pd.DataFrame()

    for u, u_items in tqdm(user_group_dfs):
        u_items = set(u_items)
        i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)
        i_user_neg_time = np.random.choice(np.arange(2005, 2015), num_negative)

        i_user_neg_df = pd.DataFrame({'user': [u] * num_negative, 'item': i_user_neg_item, 'time': i_user_neg_time,
                                      'rating': [0] * num_negative})
        if first_row == True:
            user_neg_dfs = i_user_neg_df
            first_row = False
        else:
            user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis=0, sort=False)

    user_neg_dfs = pd.merge(user_neg_dfs, genre_onehot, left_on='item', right_on='item', how='left')  # genre
    user_neg_dfs = pd.merge(user_neg_dfs, raw_year_df, left_on='item', right_on='item', how='inner')  # genre

    # 4. Join dfs
    # joined_rating_df = raw_rating_df
    joined_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis=0, sort=False)
    joined_rating_df = joined_rating_df.fillna(0)  # NaN 값 0으로 채우기


    # 5. user, item, times, years, genres을 zero-based index로 mapping
    users = list(set(joined_rating_df.loc[:, 'user']))
    users.sort()
    items = list(set((joined_rating_df.loc[:, 'item'])))
    items.sort()
    # genres = list(set((joined_rating_df.loc[:, 'genre'])))
    # genres.sort()
    times = list(set((joined_rating_df.loc[:, 'time'])))
    times.sort()
    years = list(set((joined_rating_df.loc[:, 'year'])))
    years.sort()

    # if len(users) - 1 != max(users):
    user2id = {users[i]: i for i in range(len(users))}
    joined_rating_df['user'] = joined_rating_df['user'].map(lambda x: user2id[x])
    max_time_df['user'] = max_time_df['user'].map(lambda x: user2id[x])
    users = list(set(joined_rating_df.loc[:, 'user']))

    # if len(items) - 1 != max(items):
    item2id = {items[i]: i for i in range(len(items))}
    id2item = {i: items[i] for i in range(len(items))}
    joined_rating_df['item'] = joined_rating_df['item'].map(lambda x: item2id[x])
    items = list(set((joined_rating_df.loc[:, 'item'])))

    joined_rating_df = joined_rating_df.sort_values(by=['user'])
    joined_rating_df.reset_index(drop=True, inplace=True)
    data = joined_rating_df

    n_data = len(data)
    n_user = len(users)
    n_item = len(items)
    n_genre = 18
    n_time = len(times)
    n_year = len(years)

    a_users = users * n_item
    a_items = []
    for item in items:
        a_items += [item] * n_user

    df = pd.concat((pd.DataFrame(a_users, columns=['user']), pd.DataFrame(a_items, columns=['item'])), axis=1, sort=False)
    df = pd.merge(df, max_time_df, left_on='user', right_on='user', how='left')
    id_year_df = raw_year_df.copy()
    id_year_df['item'] = id_year_df['item'].map(lambda x: item2id[x])
    df = pd.merge(df, id_year_df, left_on='item', right_on='item', how='left')  # year(개봉 연도)
    id_genre_df = genre_onehot.copy()
    id_genre_df['item'] = id_genre_df['item'].map(lambda x: item2id[x])
    df = pd.merge(df, id_genre_df, left_on='item', right_on='item', how='left')

    help_df = raw_rating_df.loc[:, ['user', 'item']]
    help_df['user'] = help_df['user'].map(lambda x: user2id[x])
    help_df['item'] = help_df['item'].map(lambda x: item2id[x])
    help_df = torch.tensor(help_df.values).to(device)

    # 6. feature matrix X, label tensor y 생성
    user_col = torch.tensor(data.loc[:, 'user']).to(device)
    item_col = torch.tensor(data.loc[:, 'item']).to(device)
    time_col = torch.tensor(data.loc[:, 'time']).to(device)
    time_min = min(time_col)
    year_col = torch.tensor(data.loc[:, 'year']).to(device)
    year_min = min(year_col)
    genre_col = torch.tensor(data.iloc[:, 4:22].values).to(device)

    offsets = [0, n_user, n_user + n_item - time_min, n_user + n_item + n_time - year_min,
               n_user + n_item + n_time + n_year]

    for col, offset in zip([user_col, item_col, time_col, year_col, genre_col], offsets):
        col += offset
    for i, j in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]):
        genre_col[:, i] += 2 * i

    X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), time_col.unsqueeze(1), year_col.unsqueeze(1),
                   genre_col], dim=1).to(device)
    y = torch.tensor(list(data.loc[:, 'rating'])).to(device)

    # 데이터 셋 형성
    dataset = RatingDataset(X, y)
    # train_ratio = 0.99
    #
    # train_size = int(train_ratio * len(data))
    # test_size = len(data) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # 평가용 데이터 셋 형성


    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # 평가
    model = DeepFM(args, [n_user, n_item, n_time, n_year, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    bce_loss = nn.BCELoss().to(device)

    for epoch in range(args.epochs):
        print('epoch :', epoch)
        for X, y in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = bce_loss(output, y.float())
            loss.backward()
            optimizer.step()

    rating_df = torch.zeros((n_user, n_item))
    rating_df = rating_df.detach().numpy()
    print(df)
    TestDS = TestDataset(df)
    print('shit')
    submission_loader = DataLoader(TestDS, batch_size=1024, shuffle=False)
    for x in tqdm(submission_loader):
        model.eval()
        output = model(x)
        for i, j in enumerate(x):
            rating_df[j[0].long()][j[1].long()] = output[i]
    for i, j in help_df:
        rating_df[i.long()][j.long()] = -np.inf
    print('rating_df', rating_df)
    submission_item = list()
    for i in tqdm(range(len(rating_df))):
        idxes = bn.argpartition(-rating_df[i], 10)[:10]  # 유저에게 추천할 10개 영화를 가져옴
        tmp = list()
        for j in idxes:
            tmp.append(id2item[j])
        submission_item.append(tmp)

    submission_item = np.array(submission_item).reshape(-1, 1)
    result = np.hstack((submission_user, submission_item))
    result = pd.DataFrame(result, columns=['user', 'item'])

    result.to_csv(os.path.join(args.output_dir, f'submission_{args.epochs}_{optimizer}.csv'), index=False)
    print("export submission : ", os.path.join(args.output_dir, f'submission_{args.epochs}_{optimizer}.csv'))


if __name__ == "__main__":
    main()
