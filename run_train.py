import argparse
import os
import time
import numpy as np
import pandas as pd
import random
from pyrsistent import v

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from importlib import import_module
from scipy import sparse
from utils import get_data, recall
from copy import deepcopy
from importlib import import_module

from models import MultiVAE, loss_function_vae, VAE
from trainers import FinetuneTrainer
import bottleneck as bn
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)

import wandb

args = None

# XXX DataLoader 클래스에 submission 추가, 경로 수정
class DataLoader():
    '''
    Load Movielens dataset
    '''
    def __init__(self, path):

        # 데이터를 불러올 폴더 지정

        if MultiVAE_args.data_process != 0:# 최근 데이터만 사용하는 경우
            self.pro_dir = os.path.join(path, f'pro_sg_{MultiVAE_args.data_process}')
        elif MultiVAE_args.data_random_process != 0:# 일정한 비율로 랜덤으로 데이터를 사용하는 경우
            self.pro_dir = os.path.join(path, f'pro_sg_random_{MultiVAE_args.data_random_process}')
        elif MultiVAE_args.train_all == True:
            self.pro_dir = os.path.join(path, 'pro_sg_all')
        else: # 기본값
            self.pro_dir = os.path.join(path, 'pro_sg')
            
            
        assert os.path.exists(self.pro_dir), "Preprocessed files do not exist. Run data.py"

        self.n_items = self.load_n_items()
    
    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        elif datatype == "submission":
            return self._load_submission_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test, submission]")
        
    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items
    
    def _load_train_data(self):
        path = os.path.join(self.pro_dir, 'train.csv')
        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data
    
    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                    (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te

    # XXX submission csv 불러오기
    # submission_data2.csv는 export_submission_data 코드를 통해 export할 수 있습니다.
    def _load_submission_data(self, datatype="submission"):
        path = '/opt/ml/input/data/eval/submission_data2.csv'
        
        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    
    return recall

def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def train(model, criterion, optimizer, is_VAE = False):
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    np.random.shuffle(idxlist)
    
    for batch_idx, start_idx in enumerate(range(0, N, MultiVAE_args.batch_size)):
        end_idx = min(start_idx + MultiVAE_args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(device)
        optimizer.zero_grad()

        if is_VAE:
          if MultiVAE_args.total_anneal_steps > 0:
            anneal = min(MultiVAE_args.anneal_cap, 
                            1. * update_count / MultiVAE_args.total_anneal_steps)
          else:
              anneal = MultiVAE_args.anneal_cap

          optimizer.zero_grad()
          recon_batch, mu, logvar = model(data)
          
          loss = criterion(recon_batch, data, mu, logvar, anneal)
        else:
          recon_batch = model(data)
          loss = criterion(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % MultiVAE_args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, MultiVAE_args.batch_size)),
                        elapsed * 1000 / MultiVAE_args.log_interval,
                        train_loss / MultiVAE_args.log_interval))
            

            start_time = time.time()
            train_loss = 0.0

# XXX evaluate 함수에 r10 추가
def evaluate(model, criterion, data_tr, data_te, is_VAE=False):
    # Turn on evaluation mode

    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n100_list = []
    r10_list = [] # r10 추가
    r20_list = []
    r50_list = []

    
    with torch.no_grad():
        for start_idx in range(0, e_N, MultiVAE_args.batch_size):
            end_idx = min(start_idx + MultiVAE_args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            device = torch.device("cuda" if MultiVAE_args.cuda else "cpu")
            data_tensor = naive_sparse2tensor(data).to(device)
            if is_VAE :
              
              if MultiVAE_args.total_anneal_steps > 0:
                  anneal = min(MultiVAE_args.anneal_cap, 
                                1. * update_count / MultiVAE_args.total_anneal_steps)
              else:
                  anneal = MultiVAE_args.anneal_cap

              recon_batch, mu, logvar = model(data_tensor)
              loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

            else :
              recon_batch = model(data_tensor)
              loss = criterion(recon_batch, data_tensor)


            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r10 = Recall_at_k_batch(recon_batch, heldout_data, 10) # r10 추가
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            r10_list.append(r10) # r10 추가
            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

    total_loss /= len(range(0, e_N, MultiVAE_args.batch_size))
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)
    r10_list = np.concatenate(r10_list) # r10 추가

    # r10 추가
    return total_loss, np.mean(n100_list), np.mean(r10_list), np.mean(r20_list), np.mean(r50_list)

# XXX submission 평가
# evaluation 함수와 비슷합니다.
def evaluate_submission(model, criterion, submission_data, is_VAE=False):
    # Turn on evaluation mode
    recon_batch_result = list()
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(submission_data.shape[0]))
    e_N = submission_data.shape[0]

    raw_data = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')

    unique_sid = pd.unique(raw_data['item'])
    unique_uid = pd.unique(raw_data['user'])
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    submission_user = list()
    submission_item = list()
    
    with torch.no_grad():
        for start_idx in range(0, e_N, MultiVAE_args.batch_size):
            end_idx = min(start_idx + MultiVAE_args.batch_size, s_N)
            data = submission_data[e_idxlist[start_idx:end_idx]]
            # true 값은 모르므로 heldout_data는 주석처리
            # heldout_data = data_te[e_idxlist[start_idx:end_idx]] 

            device = torch.device("cuda" if MultiVAE_args.cuda else "cpu")
            data_tensor = naive_sparse2tensor(data).to(device)
            if is_VAE :
              
                if MultiVAE_args.total_anneal_steps > 0:
                     anneal = min(MultiVAE_args.anneal_cap, 
                                1. * update_count / MultiVAE_args.total_anneal_steps)
                else:
                    anneal = MultiVAE_args.anneal_cap

                recon_batch, mu, logvar = model(data_tensor)
                loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

            else :
                recon_batch = model(data_tensor)
                loss = criterion(recon_batch, data_tensor)


            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf
            #recon_batch_result.append(recon_batch)
            recon_batch_result.extend(recon_batch)
            

    return np.array(recon_batch_result)


def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1

    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)

    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)

    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out

    def get_idx(self):
        return self._idx

    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)

    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]

    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)


def evaluate(model, data_in, data_out, metrics, samples_perc_per_epoch=1, batch_size=500):
    metrics = deepcopy(metrics)
    model.eval()

    for m in metrics:
        m['score'] = []

    for batch in generate(batch_size=batch_size,
                          device=device,
                          data_in=data_in,
                          data_out=data_out,
                          samples_perc_per_epoch=samples_perc_per_epoch
                         ):

        ratings_in = batch.get_ratings_to_dev()
        ratings_out = batch.get_ratings(is_out=True)

        ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()

        if not (data_in is data_out):
            ratings_pred[batch.get_ratings().nonzero()] = -np.inf

        for m in metrics:
            m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))

    for m in metrics:
        m['score'] = np.concatenate(m['score']).mean()

    return [x['score'] for x in metrics]


def run(model, opts, train_data, batch_size, n_epochs, beta, gamma, dropout_rate):
    model.train()
    for epoch in range(n_epochs):
        for batch in generate(batch_size=batch_size, device=device, data_in=train_data, shuffle=False):
            ratings = batch.get_ratings_to_dev()

            for optimizer in opts:
                optimizer.zero_grad()

            _, loss = model(ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
            loss.backward()

            for optimizer in opts:
                optimizer.step()

def result(model, data_in, samples_perc_per_epoch=1, batch_size=500):
    recon_batch_result = list()
    model.eval()
    items=[]
    user = pd.read_csv('/opt/ml/input/data/train/RecVAE/unique_uid.csv', header=None)
    item = pd.read_csv('/opt/ml/input/data/train/RecVAE/unique_sid.csv', header=None)
    item = item.to_numpy()
    for batch in generate(batch_size=batch_size,
                          device=device,
                          data_in=data_in,
                          samples_perc_per_epoch=samples_perc_per_epoch
                         ):
        
        ratings_in = batch.get_ratings_to_dev()
    
        ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()
        
        ratings_pred[batch.get_ratings().nonzero()] = -np.inf
        recon_batch_result.extend(ratings_pred)

    return np.array(recon_batch_result)

if __name__ == "__main__":
    MultiVAE_parser = argparse.ArgumentParser()
    # parser.add_argument("--model", default="S3RecModel", type=str, help="model type") # model 모듈
    MultiVAE_parser.add_argument("--model_name", default="MultiVAE", type=str)
    MultiVAE_parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    MultiVAE_parser.add_argument("--data_name", default="Ml", type=str)
    MultiVAE_parser.add_argument("--output_dir", default="/opt/ml/input/code/output/", type=str)
    MultiVAE_parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
    MultiVAE_parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    MultiVAE_parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    MultiVAE_parser.add_argument('--epochs', type=int, default=1,
                        help='upper epoch limit')
    MultiVAE_parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    MultiVAE_parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    MultiVAE_parser.add_argument('--seed', type=int, default=960708,
                        help='random seed')
    MultiVAE_parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    MultiVAE_parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    MultiVAE_parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    MultiVAE_parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")


    # XXX 추가한 args parse
    # dafault값은 모두 기존 코드와 동일한 기본값입니다.
    MultiVAE_parser.add_argument("--wandb", type=bool, default=False, help="wandb") # wandb 사용 여부
    MultiVAE_parser.add_argument('--optimizer', type=str, default='RAdam', help='optimizer type (default: Adam)') # optimizer 설정
    MultiVAE_parser.add_argument('--data_process', type=int, default=0,  help='data process') # 최근 데이터를 얼마나 사용할 것인가
    MultiVAE_parser.add_argument('--data_random_process', type=int, default=0,  help='data random process') # 데이터를 어느 비율만큼 랜덤으로 뽑을 것인가
    MultiVAE_parser.add_argument('--train_all', type=bool, default=True,  help='use all training set') # 훈련데이터를 모두 쓸 것인지
    
    MultiVAE_args = MultiVAE_parser.parse_args()

        # Set the random seed manually for reproductibility.
    torch.manual_seed(MultiVAE_args.seed)

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        MultiVAE_args.cuda = True

    device = torch.device("cuda" if MultiVAE_args.cuda else "cpu")
    print(device)

    set_seed(MultiVAE_args.seed)
    check_path(MultiVAE_args.output_dir)


    MultiVAE_args._data_file = MultiVAE_args.data_dir + "train_ratings.csv"


    RecVAE_parser = argparse.ArgumentParser()
    RecVAE_parser.add_argument('--dataset', default='/opt/ml/input/data/train/RecVAE', type=str)
    RecVAE_parser.add_argument('--hidden-dim', type=int, default=600)
    RecVAE_parser.add_argument('--latent-dim', type=int, default=300)
    RecVAE_parser.add_argument('--batch-size', type=int, default=500)
    RecVAE_parser.add_argument('--beta', type=float, default=0.4)
    RecVAE_parser.add_argument('--gamma', type=float, default=0.005)
    RecVAE_parser.add_argument('--lr', type=float, default=5e-4)
    RecVAE_parser.add_argument('--n-epochs', type=int, default=1)
    RecVAE_parser.add_argument('--n-enc_epochs', type=int, default=3)
    RecVAE_parser.add_argument('--n-dec_epochs', type=int, default=1)
    RecVAE_parser.add_argument('--not-alternating', type=bool, default=False)
    RecVAE_parser.add_argument('--optimizer', type=str, default='RAdam', help='optimizer type (default: Adam)') # optimizer 설정
    RecVAE_parser.add_argument('--wd', type=float, default=0.00,) # optimizer 설정
    
    RecVAE_args = RecVAE_parser.parse_args()

    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0")

    data = get_data(RecVAE_args.dataset)
    # train_data, test_in_data, test_out_data = data
    train_data, = data # 데이터 전체로 학습 후 결과하기 위해

    ### Multi-VAE
    ###############################################################################
    # Load data
    ###############################################################################
    
    loader = DataLoader(MultiVAE_args.data_dir)

    n_items = loader.load_n_items()
    train_data = loader.load_data('train')
    if MultiVAE_args.train_all == False:
        vad_data_tr, vad_data_te = loader.load_data('validation')
        test_data_tr, test_data_te = loader.load_data('test')

    # XXX submission data 추가
    submission_data = loader.load_data('submission')

    N = train_data.shape[0]
    s_N = submission_data.shape[0]
    idxlist = list(range(N))

    ###############################################################################
    # Build the model
    ###############################################################################

    #p_dims = [200, 600, n_items]
    p_dims = [100, n_items] # 특성을 몇개로 표현
    #p_dims = [200, 600, 800, n_items]
    #p_dims = [400, 800, n_items]
    #p_dims = [100, 300, 500, 700, n_items]
    model = MultiVAE(p_dims).to(device)

    opt_module = getattr(import_module("torch.optim"), MultiVAE_args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=MultiVAE_args.wd
    )
    criterion = loss_function_vae

    ###############################################################################
    # Training code
    ###############################################################################

    #best_n100 = -np.inf
    best_r10 = -np.inf
    update_count = 0

    # save model args
    args_str = f"{MultiVAE_args.model_name}-{MultiVAE_args.data_name}"
    MultiVAE_args.log_file = os.path.join(MultiVAE_args.output_dir, args_str + ".txt")

    # save model
    # XXX 저장 model 이름 변경
    # 여러 실험하기 위해 나누습니다
    
    if MultiVAE_args.data_process != 0 : # 최근 데이터 일부만 사용하는 경우
        # 모델이름-데이터이름_data_최근 데이터 갯수
        checkpoint = f"{args_str}_data_{MultiVAE_args.data_process}.pt"
    elif MultiVAE_args.data_random_process != 0 : # 최근 데이터 일부만 사용하는 경우
        # 모델이름-데이터이름_data_random_비율
        checkpoint = f"{args_str}_data_random_{MultiVAE_args.data_random_process}.pt"
    elif MultiVAE_args.train_all == True:
        checkpoint = f"{args_str}_data_random_train_all_100.pt"
    else : # 기본값
        # 모델이름-데이터이름_epoch 수_optimizer.pt
        checkpoint = f"{args_str}_{MultiVAE_args.epochs}_{MultiVAE_args.optimizer}.pt"
    MultiVAE_args.checkpoint_path = os.path.join(MultiVAE_args.output_dir, checkpoint)

    early_stopping = EarlyStopping(MultiVAE_args.checkpoint_path, patience=10, verbose=True)


    for epoch in range(1, MultiVAE_args.epochs + 1):
        epoch_start_time = time.time()
        train(model, criterion, optimizer, is_VAE=True) # 훈련
        if MultiVAE_args.train_all == False:
            val_loss, n100, r10, r20, r50 = evaluate(model, criterion, vad_data_tr, vad_data_te, is_VAE=True) # 검증 데이터 테스트
        if MultiVAE_args.train_all == False:
            print('-' * 100)
            print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                    'n100 {:5.3f} | r10 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                        epoch, time.time() - epoch_start_time, val_loss,
                        n100, r10, r20, r50))
            print('-' * 100)

            n_iter = epoch * len(range(0, N, MultiVAE_args.batch_size))

        # Save the model if the n100 is the best we've seen so far.
        # with open(MultiVAE_args.checkpoint_path, 'wb') as f:
        #     torch.save(model, f)
        # print(f"save model : {MultiVAE_args.checkpoint_path}")
        if MultiVAE_args.train_all == False:
            if r10 > best_r10:
                with open(MultiVAE_args.checkpoint_path, 'wb') as f:
                    torch.save(model, f)
                best_r10 = r10
                print(f"save model : {MultiVAE_args.checkpoint_path}")
        else:
            with open(MultiVAE_args.checkpoint_path, 'wb') as f:
                torch.save(model, f)
            print(f"save model : {MultiVAE_args.checkpoint_path}")

   
    if MultiVAE_args.train_all == False:
        # Load the best saved model.
        with open(MultiVAE_args.checkpoint_path, 'rb') as f:
            print(f"load model : {MultiVAE_args.checkpoint_path}")
            model = torch.load(f)       
        # Run on test data.
        test_loss, n100, r10, r20, r50 = evaluate(model, criterion, test_data_tr, test_data_te, is_VAE=True)
        print('=' * 100)
        print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r10 {:4.2f} | r20 {:4.2f} | '
                'r50 {:4.2f}'.format(test_loss, n100, r10, r20, r50))
        print('=' * 100)

    # Load the best saved model.
    with open(MultiVAE_args.checkpoint_path, 'rb') as f:
        print(f"load model : {MultiVAE_args.checkpoint_path}")
        model = torch.load(f)    

    # XXX submission 평가
    MultiVAE_result = evaluate_submission(model, criterion=loss_function_vae, submission_data=submission_data, is_VAE=True)

    ### RecVAE
    model_kwargs = {
    'hidden_dim': RecVAE_args.hidden_dim,
    'latent_dim': RecVAE_args.latent_dim,
    'input_dim': train_data.shape[1]
    }
    metrics = [{'metric': recall, 'k': 10}]

    best_recall = -np.inf
    train_scores, valid_scores = [], []

    model = VAE(**model_kwargs).to(device)
    model_best = VAE(**model_kwargs).to(device)

    learning_kwargs = {
        'model': model,
        'train_data': train_data,
        'batch_size': RecVAE_args.batch_size,
        'beta': RecVAE_args.beta,
        'gamma': RecVAE_args.gamma
    }

    decoder_params = set(model.decoder.parameters())
    encoder_params = set(model.encoder.parameters())

    opt_encoder_module = getattr(import_module("torch.optim"), RecVAE_args.optimizer)  # default: Adam
    opt_decoder_module = getattr(import_module("torch.optim"), RecVAE_args.optimizer)  # default: Adam
    optimizer_encoder = opt_encoder_module(
            encoder_params,
            lr=RecVAE_args.lr,
            weight_decay=RecVAE_args.wd
        )

    optimizer_decoder = opt_decoder_module(
            decoder_params,
            lr=RecVAE_args.lr,
            weight_decay=RecVAE_args.wd
        )

    for epoch in range(RecVAE_args.n_epochs):
        if RecVAE_args.not_alternating:
            run(opts=[optimizer_encoder, optimizer_decoder], n_epochs=1, dropout_rate=0.5, **learning_kwargs)
        else:
            run(opts=[optimizer_encoder], n_epochs=RecVAE_args.n_enc_epochs, dropout_rate=0.5, **learning_kwargs)
            model.update_prior()
            run(opts=[optimizer_decoder], n_epochs=RecVAE_args.n_dec_epochs, dropout_rate=0, **learning_kwargs)

        train_scores.append(
            evaluate(model, train_data, train_data, metrics, 0.01)[0]
        )

        #wandb.log({'score': train_scores[-1]})
        
        # if train_scores[-1] > best_recall:
        #     best_recall = train_scores[-1]
        #     model_best.load_state_dict(deepcopy(model.state_dict()))
        #     print("save RecVAE model")
        model_best.load_state_dict(deepcopy(model.state_dict()))
        print("save RecVAE model")
            

        print(f'epoch {epoch} | train recall@10: {train_scores[-1]:.4f}')

    RecVAE_result = result(model_best,train_data)

    ensemble_result = RecVAE_result#+MultiVAE_result*0.2

    # XXX submission 결과
    raw_data = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
    submission_user = list()
    submission_item = list()



    unique_sid = pd.unique(raw_data['item'])
    unique_uid = pd.unique(raw_data['user'])
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    for i in range(len(ensemble_result)):
        idxes = bn.argpartition(-ensemble_result[i], 10)[:10] # 유저에게 추천할 10개 영화를 가져옴
        tmp = list()
        # id2show 과정
        for j in idxes:                    
            tmp.append(list(show2id.keys())[j]) # id2show # = tmp.append(raw_data['item'].unique()[j])
        submission_item.append(tmp)

    submission_item = np.array(submission_item).reshape(-1, 1)
    submission_user = raw_data['user'].unique().repeat(10)
    submission_user = np.array(submission_user).reshape(-1, 1)

    submission_result = np.hstack((submission_user,submission_item))

    submission_result = pd.DataFrame(submission_result, columns=['user','item'])  
    submission_result.to_csv(os.path.join(MultiVAE_args.output_dir, f'submission_ensemble.csv'), index=False)
    print("export submission : ", os.path.join(MultiVAE_args.output_dir, f'submission_ensemble.csv'))

        # XXX submission export 
    # if MultiVAE_args.data_process != 0: # 최근 데이터 일부만 사용한 경우
    #     # submission_data_최근 데이터 갯수.csv
    #     result.to_csv(os.path.join(MultiVAE_args.output_dir, f'submission_data_{MultiVAE_args.data_process}.csv'), index=False)
    #     print("export submission : ", os.path.join(MultiVAE_args.output_dir, f'submission_data_{MultiVAE_args.data_process}.csv'))
    # elif MultiVAE_args.data_random_process != 0:
    #     result.to_csv(os.path.join(MultiVAE_args.output_dir, f'submission_data_random_{MultiVAE_args.data_random_process}.csv'), index=False)
    #     print("export submission : ", os.path.join(MultiVAE_args.output_dir, f'submission_data_random_{MultiVAE_args.data_random_process}.csv'))
    # elif MultiVAE_args.train_all == True:
    #     result.to_csv(os.path.join(MultiVAE_args.output_dir, f'submission_data_all_{MultiVAE_args.batch_size}_100.csv'), index=False)
    #     print("export submission : ", os.path.join(MultiVAE_args.output_dir, f'submission_data_all_{MultiVAE_args.batch_size}_100.csv'))
    # else : # 기본값             
    #     # submission_epoch 수_optimizer.csv
    #     result.to_csv(os.path.join(MultiVAE_args.output_dir, f'submission_{MultiVAE_args.epochs}_{MultiVAE_args.optimizer}.csv'), index=False)
    #     print("export submission : ", os.path.join(MultiVAE_args.output_dir, f'submission_{MultiVAE_args.epochs}_{MultiVAE_args.optimizer}.csv'))
    

