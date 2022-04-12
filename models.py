import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class EASE:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'user'])
        items = self.item_enc.fit_transform(df.loc[:, 'item'])
        return users, items

    def fit(self, df, lambda_: float = 0.5, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = self._get_users_and_items(df)
        values = np.ones(df.shape[0]) if implicit else df['rating'].to_numpy() / df['rating'].max()

        X = csr_matrix((values, (users, items)))
        self.X = X
        print("finish X matrix")
        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)
        print("finish B matrix")

    def predict(self, train, users, items, k):
        df = pd.DataFrame()
        items = self.item_enc.transform(items)
        dd = train.loc[train.user.isin(users)]
        dd['ci'] = self.item_enc.transform(dd.item)
        dd['cu'] = self.user_enc.transform(dd.user)

        g = dd.groupby('user')
        for user, group in tqdm(g):
            watched = set(group['ci']) # 본 영화

            # 본 영화를 item 중에 제거한 index list.
            candidates = [item for item in items if item not in watched]
            u = group['cu'].iloc[0] # 예측한 user 번호
            pred = np.take(self.pred[u, :], candidates) # 유저의 예측 값중 후보들만 가져온다.
            res = np.argpartition(pred, -k)[-k:] # Top 10
            r = pd.DataFrame({
                "user": [user] * len(res),
                "item": np.take(candidates, res),
                "score": np.take(pred, res)
            }).sort_values('score', ascending=False)
            df = df.append(r, ignore_index=True)
        df['item'] = self.item_enc.inverse_transform(df['item'])
        print("finish predict")
        return df

    def make_all_predicted(self):
        all_ = pd.DataFrame(self.pred)
        all_.to_csv(
            "/workspace/output/all_result.csv", index=False
        )


# using year
class EASE2:
    def __init__(self, train):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
        self.last_interaction = LabelEncoder()
        self.item_year = LabelEncoder()

        self.train = train

        self.item_year_df = train[['item', 'item_year']]
        self.user_last_interaction = train[['user', 'last_interaction']]

        item_year_list = [1920, 1921, 1916, 1917, 1902, 1915, 2015, 1919]
        for index, item in enumerate([6987, 3310, 7243, 8511, 32898, 7065, 119145, 6988]):
            self.item_year_df.loc[self.item_year_df.item == item, ('item_year')] = (item_year_list[index])

        self.item_year_df.drop_duplicates(['item','item_year'])
        self.user_last_interaction.drop_duplicates(['user','last_interaction'])
        self.item_year_dict = self.item_year_df.set_index('item').to_dict()['item_year']
        self.user_last_interaction_dict = self.user_last_interaction.set_index('user').to_dict()['last_interaction']

        print("init finished")

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'user'])
        items = self.item_enc.fit_transform(df.loc[:, 'item'])
        return users, items

    def fit(self, df, lambda_: float = 0.5, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = self._get_users_and_items(df)
        values = np.ones(df.shape[0]) if implicit else df['rating'].to_numpy() / df['rating'].max()

        X = csr_matrix((values, (users, items)))
        self.X = X
        print("finish X matrix")
        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)
        print("finish B matrix")

    def predict(self, train, users, items, k):
        df = pd.DataFrame()
        items = self.item_enc.transform(items)
        dd = train.loc[train.user.isin(users)]
        dd['ci'] = self.item_enc.transform(dd.item)
        dd['cu'] = self.user_enc.transform(dd.user)

        g = dd.groupby('user')
        for user, group in tqdm(g):
            # wandb.log({'user': user})
            watched = set(group['ci']) # 본 영화

            # 본 영화를 item 중에 제거한 index list.
            candidates = [item for item in items if item not in watched]
            last_interaction = self.user_last_interaction_dict[user]
            # last_interaction = \
            # self.user_last_interaction[self.user_last_interaction['user'] == user]['last_interaction'].iloc[0]
            will_be_removed = []

            original_cand = self.item_enc.inverse_transform(candidates)

            for index, movie_number in enumerate(original_cand):
                movie_year = self.item_year_dict[movie_number]
                # movie_year = self.item_year_df[self.item_year_df['item'] == movie_number]['item_year'].iloc[0]
                if int(movie_year) > last_interaction:
                    will_be_removed.append(index)

            final_cand = np.delete(candidates, will_be_removed)

            u = group['cu'].iloc[0] # 예측한 user 번호
            pred = np.take(self.pred[u, :], final_cand) # 유저의 예측 값중 후보들만 가져온다.
            res = np.argpartition(pred, -k)[-k:] # Top 10
            r = pd.DataFrame({
                "user": [user] * len(res),
                "item": np.take(final_cand, res),
                "score": np.take(pred, res)
            }).sort_values('score', ascending=False)
            df = df.append(r, ignore_index=True)

        df['item'] = self.item_enc.inverse_transform(df['item'])
        print("finish predict")
        return df