import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--p', default=0.2 , type=float)
args = parser.parse_args()
p = args.p

i_u = pd.read_csv('../i_u.csv')

train=[[] for _ in range(len(i_u))]
test=[[] for _ in range(len(i_u))]
for i in range(len(i_u)):
    items=i_u.iloc[i]
    items = np.array(items)
    l = len(items)
    idx = np.zeros(l, dtype='bool')
    idx[np.random.choice(l,int(l*p),replace=False)]=True
    train[i]=items[np.logical_not(idx)]
    test[i]=items[idx]

train = pd.DataFrame(train)
test = pd.DataFrame(test)

train.to_csv('train.csv', index = False)
test.to_csv('test.csv', index = False)

