# based on https://github.com/dawenl/vae_cf

import os
import sys

import numpy as np
from scipy import sparse
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--min_users_per_item', type=int, default=0)
parser.add_argument('--heldout_users', type=int)

args = parser.parse_args()

dataset = args.dataset
output_dir = args.output_dir
min_sc = args.min_users_per_item
n_heldout_users = args.heldout_users

raw_data = pd.read_csv(dataset, header=0)

unique_uid = raw_data['user'].unique()

# np.random.seed(98765)
# idx_perm = np.random.permutation(unique_uid.size)
# unique_uid = unique_uid[idx_perm]

n_users = unique_uid.size

tr_users = unique_uid[:(n_users - n_heldout_users)]
# te_users = unique_uid[(n_users - n_heldout_users):]

train_plays = raw_data.loc[raw_data['user'].isin(tr_users)]

unique_sid = pd.unique(train_plays['item'])

show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)
        
with open(os.path.join(output_dir, 'unique_uid.txt'), 'w') as f:
    for uid in unique_uid:
        f.write('%s\n' % uid)


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        idx = np.zeros(n_items_u, dtype='bool')
        idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

        tr_list.append(group[np.logical_not(idx)])
        te_list.append(group[idx])


        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    
    return data_tr, data_te


# test_plays = raw_data.loc[raw_data['user'].isin(te_users)]
# test_plays = test_plays.loc[test_plays['item'].isin(unique_sid)]

# test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

def numerize(tp):
    uid = list(map(lambda x: profile2id[x], tp['user']))
    sid = list(map(lambda x: show2id[x], tp['item']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


train_data = numerize(train_plays)
train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

# test_data_tr = numerize(test_plays_tr)
# test_data_tr.to_csv(os.path.join(output_dir, 'test_tr.csv'), index=False)

# test_data_te = numerize(test_plays_te)
# test_data_te.to_csv(os.path.join(output_dir, 'test_te.csv'), index=False)