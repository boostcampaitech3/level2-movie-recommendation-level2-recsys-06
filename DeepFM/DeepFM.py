import argparse
import os

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
    def __init__(self, args, input_dims): # input_dims = [n_user, n_item, n_time, n_year, n_genre]
        super(DeepFM, self).__init__()
        self.mlp_dims = args.mlp_dims # [200, 200, 200]
        self.drop_rate = args.dropout # 0.5

        total_input_dim = int(sum(input_dims))  # n_user + n_movie + n_time + n_year + n_genre

        # Fm component의 constant bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = nn.Embedding(total_input_dim, 1) # (n_user + n_movie + n_time + n_year + n_genre) -> 1

        self.embedding = nn.Embedding(total_input_dim, args.embedding_dim) # (n_user + n_movie + n_time + n_year + n_genre) -> 10
        self.embedding_dim = len(input_dims) * args.embedding_dim # 5 * 10

        mlp_layers = []
        for i, dim in enumerate(self.mlp_dims):
            if i == 0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim)) # Linear 5 * 10 -> 200
            else:
                mlp_layers.append(nn.Linear(self.mlp_dims[i - 1], dim)) # Linear 200 -> 200
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(self.drop_rate))
        mlp_layers.append(nn.Linear(self.mlp_dims[-1], 1)) # Linear 200 -> 1
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
        name="DEEPFM_negative150_layer300_epoch11",  # -> str : ex) "모델_파라티머_파라미터_파라미터", 훈련 정보에 대해 알아보기 쉽게
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
        "--mlp_dims", type=list, default=[300, 300, 300], help="define mlp layers"
    )
    # 몇 층으로 구성 되었는 지,
    # network shape (constant/increasing/decreasing/diamond),
    # 층 당 뉴런의 개수를 정의
    parser.add_argument("--negative", default=150, type=int)  # Negative instance 생성 개수
    parser.add_argument("--activation", default="relu", type=str)  # gelu relu
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="hidden dropout p"
    )

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=11, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    # parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # parser.add_argument(
    #     "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    # )
    # parser.add_argument(
    #     "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    # )
    # parser.add_argument(
    #     "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    # )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()
    wandb.config.update(args)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    device = torch.device("cuda:0")  # TODO cuda

    # process 1 데이터 파일 로드 및 구성
    raw_rating_df = pd.read_csv(args.data_dir + 'train_ratings.csv') # user_num | item_num
    raw_rating_df['time'] = pd.to_datetime(raw_rating_df['time'], unit='s').dt.year  # time(시청 연도)
    # user_num | item_num | time
    max_time_df = raw_rating_df.groupby(['user'])['time'].max() # 유저별 시청 연도의 최대값
    max_time_df = pd.DataFrame(max_time_df).reset_index() # 데이터 프레임으로 변환
    # max_time_df : user_num | item_num | max_time
    # max_time_df = max_time_df.reset_index()

    # year 결측치
    raw_year_df = pd.read_csv("../../data/train/years.tsv", sep='\t') # year 파일 불러오기
    year_df = pd.DataFrame({'item': [6987, 3310, 7243, 8511, 32898, 7065, 119145, 6988],
                            'year': [1920, 1921, 1916, 1917, 1902, 1915, 2015, 1919]})
    raw_year_df = pd.concat([raw_year_df, year_df], axis=0, sort=False)
    raw_rating_df = pd.merge(raw_rating_df, raw_year_df, left_on='item', right_on='item', how='inner')  # year(개봉 연도)
    # user_num | item_num | time | year

    help_df = raw_rating_df.loc[:, ['user', 'item']] # 이미 시청한 영화 목록 # user_num | item_num

    raw_genre_df = pd.read_csv("../../data/train/genres.tsv", sep='\t') # 장르 데이터
    genre_dict = {genre: i for i, genre in enumerate(set(raw_genre_df['genre']))} # {genre_num : genre_index}
    raw_genre_df['genre'] = raw_genre_df['genre'].map(lambda x: genre_dict[x]) # genre number -> genre_index
    raw_rating_df = pd.merge(raw_rating_df, raw_genre_df, left_on='item', right_on='item', how='left')
    # user_num | item_num | time | year | genre_id

    raw_rating_df['rating'] = 1.0  # implicit feedback
    # user_num | item_num | time | year | genre_id | rating(1)

    itemset = set(raw_rating_df.loc[:, 'item'])  # item_num 집합

    # 3. Negative instance 생성
    print("Create Negative instances")
    num_negative = args.negative
    user_group_dfs = list(raw_rating_df.groupby('user')['item']) # user_num, item_num 으로 groupby
    first_row = True
    user_neg_dfs = pd.DataFrame()

    for u, u_items in tqdm(user_group_dfs):
        u_items = set(u_items)
        i_user_neg_item = np.random.choice(list(itemset - u_items), num_negative, replace=False) # 선택하지 않은 아이템 중에서 num_negative 개수만큼 선택
        i_user_neg_time = np.random.choice(np.arange(2005, 2015), num_negative) # 2005-2015년 중에서 num_negative 개수만큼 선택

        i_user_neg_df = pd.DataFrame({'user': [u] * num_negative, 'item': i_user_neg_item, 'time': i_user_neg_time,
                                      'rating': [0] * num_negative}) # user_num | neg_item | neg_time | implicit feedback
        if first_row == True:
            user_neg_dfs = i_user_neg_df
            first_row = False
        else:
            user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis=0, sort=False)

    user_neg_dfs = pd.merge(user_neg_dfs, raw_genre_df, left_on='item', right_on='item', how='left')  # genre
    user_neg_dfs = pd.merge(user_neg_dfs, raw_year_df, left_on='item', right_on='item', how='inner')  # year
    # user_num | neg_item | neg_time | rating(0) | genre_index | year

    # 4. Join dfs
    joined_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis=0, sort=False)
    # joined_rating_df : user_num | item_num | time | genre_index | year | rating
    # joined_rating_df = joined_rating_df.fillna(0)  # NaN 값 0으로 채우기
    print('joined_rating_df')

    # 5. user, item, times, years, genres을 zero-based index로 mapping
    users = list(set(joined_rating_df.loc[:, 'user']))
    users.sort()
    items = list(itemset)
    items.sort()
    genres = list(set((joined_rating_df.loc[:, 'genre'])))
    # genres.sort()
    times = list(set((joined_rating_df.loc[:, 'time'])))
    # times.sort()
    years = list(set((joined_rating_df.loc[:, 'year'])))
    # years.sort()

    # if len(users) - 1 != max(users):
    user2id = {users[i]: i for i in range(len(users))} # user_num -> user_index
    id2user = {i: users[i] for i in range(len(users))} # user_id -> user_num
    joined_rating_df['user'] = joined_rating_df['user'].map(lambda x: user2id[x])
    # joined_rating_df : user_id | neg_item | neg_time | genre_index | year| rating
    max_time_df['user'] = max_time_df['user'].map(lambda x: user2id[x])
    # max_time_df : user_id | item_num | max_time

    # if len(items) - 1 != max(items):
    item2id = {items[i]: i for i in range(len(items))} # item_num -> item_index
    id2item = {i: items[i] for i in range(len(items))} # item_index -> item_num
    joined_rating_df['item'] = joined_rating_df['item'].map(lambda x: item2id[x])
    # joined_rating_df : user_id | item_id | time | year | genre_index | rating

    joined_rating_df = joined_rating_df.sort_values(by=['user'])
    joined_rating_df.reset_index(drop=True, inplace=True)
    # joined_rating_df : user_id(sorted) | item_id | time | genre_index | year | rating
    data = joined_rating_df
    # data : user_id(sorted) | item_id | time | genre_index | year | rating

    n_user = len(users)
    n_item = len(items)
    n_genre = len(genres)
    n_time = len(times)
    n_year = len(years)

    a_users = [i for i in range(n_user)] * n_item  # [user_id] * n_item(6807)
    a_items = [] # [[item_id * n_user]]
    for item in range(n_item):
        a_items += [item] * n_user

    df = pd.concat((pd.DataFrame(a_users, columns=['user']), pd.DataFrame(a_items, columns=['item'])), axis=1,
                   sort=False)
    # df : user_id | item_id (item 기준으로 묶임)
    df = pd.merge(df, max_time_df, left_on='user', right_on='user', how='left')
    # df : user_id | item_id | max_time
    id_year_df = raw_year_df.copy()
    # item_num | year
    id_year_df['item'] = id_year_df['item'].map(lambda x: item2id[x])
    # item_id | year
    df = pd.merge(df, id_year_df, left_on='item', right_on='item', how='left')  # year(개봉 연도)
    # df : user_id | item_id | max_time | year
    id_genre_df = raw_genre_df.copy()
    # item_num | genre_id
    id_genre_df['item'] = id_genre_df['item'].map(lambda x: item2id[x])
    # item_id | genre_id
    df = pd.merge(df, id_genre_df, left_on='item', right_on='item', how='left')
    # df : user_id | item_id | max_time | year | genre_id
    print('make df')

    # help_df : user_num | item_num
    help_df['user'] = help_df['user'].map(lambda x: user2id[x])
    help_df['item'] = help_df['item'].map(lambda x: item2id[x])
    help_df['rating'] = -np.inf
    # help_df : user_id | item_id | rating(-inf) # 이미 시청한 영화 목록 # 이미 본 영화는 -inf로 지정
    print('make help_df')
    print('help_df\n', help_df)

    # 6. offset 추가하고 feature matrix X, label tensor y 생성
    user_col = torch.tensor(data.loc[:, 'user'])
    item_col = torch.tensor(data.loc[:, 'item'])
    time_col = torch.tensor(data.loc[:, 'time'])
    time_min = min(time_col)
    year_col = torch.tensor(data.loc[:, 'year'])
    year_min = min(year_col)
    genre_col = torch.tensor(data.loc[:, 'genre'])

    offsets = [0, n_user, n_user + n_item - time_min, n_user + n_item + n_time - year_min,
               n_user + n_item + n_time + n_year]

    for col, offset in zip([user_col, item_col, time_col, year_col, genre_col], offsets):
        col += offset

    X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), time_col.unsqueeze(1), year_col.unsqueeze(1),
                   genre_col.unsqueeze(1)], dim=1).to(device)
    # X : user_id(sorted) | item_id | time | year | genre_index
    y = torch.tensor(list(data.loc[:, 'rating'])).to(device)
    # y : rating

    # 데이터 셋 형성
    print('make dataset')
    dataset = RatingDataset(X, y)
    # train_ratio = 0.99
    #
    # train_size = int(train_ratio * len(data))
    # test_size = len(data) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # save model args
    # args_str = f"{args.model_name}-{args.data_name}"
    # args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    # print(str(args))

    # save model
    # checkpoint = args_str + ".pt"
    # args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # 평가
    model = DeepFM(args, [n_user, n_item, n_time, n_year, n_genre]).to(device)
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
        wandb.log({
            'loss':loss,
        })
        # negative sampling user 하나당 50개 씩으로 했다. ~ 더 늘려볼까?

    print(df)
    TestDS = TestDataset(df)
    submission_loader = DataLoader(TestDS, batch_size=1024, shuffle=False)
    help_rating_df = torch.tensor([]).to(device)
    model.eval()

    with torch.no_grad():
        for x in tqdm(submission_loader):
            output = model(x)
            help_rating_df = torch.cat((help_rating_df, output), 0).to(device)
    # help_rating_df : calculated_rating
    help_rating_df = pd.DataFrame(help_rating_df.cpu())
    print(help_rating_df)
    rating_df = pd.concat([df, help_rating_df], axis=1)
    # rating_df : user_id | item_id | max_time | year | genre_id | calculated_rating
    rating_df.drop(['time', 'year', 'genre'], axis=1, inplace=True)
    rating_df.columns = ['user', 'item', 'rating']
    # rating_df : user_id | item_id | calculated_rating (genre_id때문에 동일한 item_id에도 여러 rating이 존재)
    rating_df = rating_df.groupby(['user', 'item']).max().reset_index()
    # rating_df : user_id | item_id | calculated_rating (max값만 남겼기 때문에 item_id당 하나의 rating만 존재)
    print(rating_df)

    rating_df = pd.concat([rating_df, help_df], axis=0)
    # rating_df : user_id | item_id | calculated_rating + (-inf) (이미 시청한 영화는 calculated_rating & -inf 값이 존재)
    rating_df = rating_df.groupby(['user', 'item']).min().reset_index()
    # rating_df : user_id | item_id | calculated_rating (min 값을 취함으로써 이미 시청한 영화는 -inf, 그렇지 않은 영화는 calc_rating 존재)
    print('rating_df1\n', rating_df)
    rating_df = rating_df.sort_values(['user', 'rating'], ascending=[True, False])
    # rating_df : user_id(오름차순) | item_id | calculated_rating(내림차순) (유저에 대해서 오름차순/ 점수에 대해서 내림차순 정렬)
    rating_df.reset_index(drop=True, inplace=True)
    print('rating_df2\n', rating_df)

    submission_user = []
    for i in range(n_user):
        submission_user += [i] * 10
    submission_user = np.array(submission_user).reshape(-1, 1)
    # submission_user : [user_id i * 10] array

    submission_item = list()
    for i in tqdm(range(n_user)):
        top_ten_id = rating_df.iloc[i*n_item:i*n_item + 10]['item']
        # user_id가 i 번인 유저의 top10 item_id DataFrame을 추출
        top_ten_id = top_ten_id.values.tolist()
        submission_item.append(top_ten_id)
    submission_item = np.array(submission_item).reshape(-1, 1)
    # submission_item : [user_id i's top 10 item_id]

    result = np.hstack((submission_user, submission_item))
    # result : [user_id i & top 10 item_id]
    result = pd.DataFrame(result, columns=['user', 'item'])
    result['user'] = result['user'].map(lambda x: id2user[x])
    result['item'] = result['item'].map(lambda x: id2item[x])
    # result : [user_num & top 10 item_num]

    result.to_csv(os.path.join(args.output_dir, f'submission_{args.epochs}_{optimizer}.csv'), index=False)
    print("export submission : ", os.path.join(args.output_dir, f'submission_{args.epochs}_{optimizer}.csv'))


if __name__ == "__main__":
    main()
