import argparse
import os
import time
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from importlib import import_module
from scipy import sparse

from models import MultiVAE, loss_function_vae
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

class DataLoader():
    '''
    Load Movielens dataset
    '''
    def __init__(self, path):
        
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

    def _load_submission_data(self, datatype="submission"):
        path = os.path.join(self.pro_dir, '{}_data2.csv'.format(datatype))
        
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
    
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(device)
        optimizer.zero_grad()

        if is_VAE:
          if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap, 
                            1. * update_count / args.total_anneal_steps)
          else:
              anneal = args.anneal_cap

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

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval,
                        train_loss / args.log_interval))
            

            start_time = time.time()
            train_loss = 0.0

def evaluate(model, criterion, data_tr, data_te, is_VAE=False):
    # Turn on evaluation mode

    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n100_list = []
    r10_list = []
    r20_list = []
    r50_list = []

    
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            device = torch.device("cuda" if args.cuda else "cpu")
            data_tensor = naive_sparse2tensor(data).to(device)
            if is_VAE :
              
              if args.total_anneal_steps > 0:
                  anneal = min(args.anneal_cap, 
                                1. * update_count / args.total_anneal_steps)
              else:
                  anneal = args.anneal_cap

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
            r10 = Recall_at_k_batch(recon_batch, heldout_data, 10)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            r10_list.append(r10)
            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

    total_loss /= len(range(0, e_N, args.batch_size))
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)
    r10_list = np.concatenate(r10_list)

    return total_loss, np.mean(n100_list), np.mean(r10_list), np.mean(r20_list), np.mean(r50_list)

def evaluate_submission(model, criterion, submission_data, is_VAE=False):
    # Turn on evaluation mode

    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(submission_data.shape[0]))
    e_N = submission_data.shape[0]

    raw_data = pd.read_csv(os.path.join(args.data_dir)+'train_ratings.csv')

    unique_sid = pd.unique(raw_data['item'])
    unique_uid = pd.unique(raw_data['user'])
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    submission_user = list()
    submission_item = list()
    count = 0
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, s_N)
            data = submission_data[e_idxlist[start_idx:end_idx]]
            # heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            device = torch.device("cuda" if args.cuda else "cpu")
            data_tensor = naive_sparse2tensor(data).to(device)
            if is_VAE :
              
              if args.total_anneal_steps > 0:
                  anneal = min(args.anneal_cap, 
                                1. * update_count / args.total_anneal_steps)
              else:
                  anneal = args.anneal_cap

              recon_batch, mu, logvar = model(data_tensor)

              loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

            else :
              recon_batch = model(data_tensor)
              loss = criterion(recon_batch, data_tensor)


            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            # FIXME    
            # 데이터 전처리 과정에서 show2id
            # 따라서 결과를 뽑을 때는 다시 id2show해야함
            # show2id는 원본 데이터 순서로 dict형태 (show(원본 영화 id), id(0번부터))
            # 따라서 show2id id번째의 key값이 원래 영화 id
            for i in range(len(recon_batch)):
                idxes = bn.argpartition(-recon_batch[i], 10)[:10] # 유저에게 추천할 10개 영화를 가져옴
                tmp = list()
                count += 1
                # id2show 과정
                for j in idxes:                    
                    tmp.append(list(show2id.keys())[j]) # id2show # = tmp.append(raw_data['item'].unique()[j])
                submission_item.append(tmp)

    submission_item = np.array(submission_item).reshape(-1, 1)
    submission_user = raw_data['user'].unique().repeat(10)
    submission_user = np.array(submission_user).reshape(-1, 1)

    result = np.hstack((submission_user,submission_item))

    result = pd.DataFrame(result, columns=['user','item'])
    result.to_csv(os.path.join(args.output_dir, 'submission_test.csv'), index=False)

    print("export submission done!")

def main():
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
        args.data_file
    )

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # if args.using_pretrain:
    #     pretrained_path = os.path.join(args.output_dir, {args.using_pretrain_model_name}+'.pt')
    #     try:
    #         trainer.load(pretrained_path)
    #         print(f"Load Checkpoint From {pretrained_path}!")

    #     except FileNotFoundError:
    #         print(f"{pretrained_path} Not Found! The Model is same as SASRec")
    # else:
    #     print("Not using pretrained model. The Model is same as SASRec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", default="S3RecModel", type=str, help="model type") # model 모듈
    parser.add_argument("--model_name", default="MultiVAE", type=str)
    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--wandb", type=bool, default=False, help="wandb")
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    
    args = parser.parse_args()

    # Set the random seed manually for reproductibility.
    torch.manual_seed(args.seed)

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    set_seed(args.seed)
    check_path(args.output_dir)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

    # wandb
    if args.wandb:
        wandb.login()
        wandb.init(group="Multi-VAE", project="MovieLens", entity="recsys-06", name=f"Multi-VAE_{args.epochs}_{args.optimizer}")
        wandb.config = args

    ###############################################################################
    # Load data
    ###############################################################################

    loader = DataLoader(args.data_dir)

    n_items = loader.load_n_items()
    train_data = loader.load_data('train')
    vad_data_tr, vad_data_te = loader.load_data('validation')
    test_data_tr, test_data_te = loader.load_data('test')
    submission_data = loader.load_data('submission')

    N = train_data.shape[0]
    s_N = submission_data.shape[0]
    idxlist = list(range(N))

    ###############################################################################
    # Build the model
    ###############################################################################

    p_dims = [200, 600, n_items]
    model = MultiVAE(p_dims).to(device)

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=args.wd
    )
    criterion = loss_function_vae

    ###############################################################################
    # Training code
    ###############################################################################

    #best_n100 = -np.inf
    best_r20 = -np.inf
    update_count = 0


    # trainer = FinetuneTrainer(
    #     model, train_dataloader, eval_dataloader, test_dataloader, None, args
    # )

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")

    # args.item2attribute = item2attribute
    # # set item score in train set to `0` in validation
    # args.train_matrix = valid_rating_matrix


    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(model, criterion, optimizer, is_VAE=True)
        val_loss, n100, r10, r20, r50 = evaluate(model, criterion, vad_data_tr, vad_data_te, is_VAE=True)
        if args.wandb:
            wandb.log({
                "val_loss" : val_loss,
                "n100" : n100,
                "r10" : r10,
                "r20" : r20,
                "r50" : r50
            })
        print('-' * 100)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                'n100 {:5.3f} | r10 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                    epoch, time.time() - epoch_start_time, val_loss,
                    n100, r10, r20, r50))
        print('-' * 100)

        n_iter = epoch * len(range(0, N, args.batch_size))

        # Save the model if the n100 is the best we've seen so far.
        if r20 > best_r20:
            with open(args.checkpoint_path, 'wb') as f:
                torch.save(model, f)
            best_r20 = n100
            print(f"save model : {args.checkpoint_path}")

    # Load the best saved model.
    with open(args.checkpoint_path, 'rb') as f:
        print(f"load model : {args.checkpoint_path}")
        model = torch.load(f)       

    
    # Run on test data.
    test_loss, n100, r10, r20, r50 = evaluate(model, criterion, test_data_tr, test_data_te, is_VAE=True)
    print('=' * 100)
    print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r10 {:4.2f} | r20 {:4.2f} | '
            'r50 {:4.2f}'.format(test_loss, n100, r10, r20, r50))
    print('=' * 100)


    evaluate_submission(model, criterion=loss_function_vae, submission_data=loader.load_data("submission"), is_VAE=True)
