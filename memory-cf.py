import pandas as pd
import os
import numpy as np
from tqdm import tqdm

#불러오기
path = './input/data/train/'
rating = pd.read_csv(os.path.join(path)+'train_ratings.csv')

#matrix 만들기
m=np.zeros((len(rating['user'].unique()),len(rating['item'].unique())))
m = pd.DataFrame(m)
m = m.set_index(keys=rating['user'].unique())
m.columns = rating['item'].unique()
for i in range(rating.shape[0]):
    user=rating.iloc[i]['user']
    item=rating.iloc[i]['item']
    m.loc[user][item] = 1
m.to_csv('matrix.csv')

#jaccard 구하기
jaccard=np.zeros((6807,6087))
jaccard=pd.DataFrame(jaccard)

user_i=[]
for i in range(6807):
    user_i.append(np.where(m[i]==1)[0])

n_user_i=[]
for i in range(6807):
    n_user_i.append(len(user_i[i]))

for i in tqdm(range(6806)):
    for j in range(i+1,6807):
        n_inter = len(set(user_i[i])&set(user_i[j]))
        n_union = n_user_i[i]+n_user_i[j]-n_inter
        jaccard[i][j] = n_inter/n_union
jaccard.to_csv('jaccard.csv')

j=jaccard.to_numpy()
j+=j.T
for i in range(6807):
    j[i][i] = -6807
jaccard=pd.DataFrame(j)
jaccard.to_csv('jaccard2.csv')

jaccard.columns = rating['item'].unique()

# user별 item 구하기
item_u=[]
for i in range(31360):
    item_u.append(np.where(m.iloc[i]==1)[0])
item_u

# 추천 item 구하기
items=[]
itemnumber=rating['item'].unique()
for item in tqdm(item_u):
    idx=np.argpartition(jaccard[item].sum(axis=1),-10)[-10:]
    items.append(itemnumber[idx])

# 결과 구하기
items=np.array(items).reshape(-1,1)
users = rating['user'].unique().repeat(10)
users = np.array(users).reshape(-1,1)

result = np.concatenate(users,items,axis=1)
result = pd.DataFrame(result, columns=['user','item'])
result.to_csv('result.csv', index=False)
