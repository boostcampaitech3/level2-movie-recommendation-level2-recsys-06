import argparse
import os
import time
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from importlib import import_module
from scipy import sparse

from models import MultiVAE, loss_function_vae, focal_loss
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

        if args.data_process != 0:# 최근 데이터만 사용하는 경우
            self.pro_dir = os.path.join(path, f'pro_sg_{args.data_process}')
        elif args.data_random_process != 0:# 일정한 비율로 랜덤으로 데이터를 사용하는 경우
            self.pro_dir = os.path.join(path, f'pro_sg_random_{args.data_random_process}')
        elif args.train_all == True:
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
          
          # loss = criterion(recon_batch, data, mu, logvar, anneal)
          loss = criterion(recon_batch, data)
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
              #loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
              loss = criterion(recon_batch, data)

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

    total_loss /= len(range(0, e_N, args.batch_size))
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
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, s_N)
            data = submission_data[e_idxlist[start_idx:end_idx]]
            # true 값은 모르므로 heldout_data는 주석처리
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
                #loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
                loss = criterion(recon_batch, data_tensor)

            else :
                recon_batch = model(data_tensor)
                loss = criterion(recon_batch, data_tensor)


            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf
            
            # XXX submission 결과
            # 데이터 전처리 과정에서 show2id
            # 따라서 결과를 뽑을 때는 다시 id2show해야함
            # show2id는 원본 데이터 순서로 dict형태 (show(원본 영화 id), id(0번부터))
            # 따라서 show2id id번째의 key값이 원래 영화 id
            for i in range(len(recon_batch)):
                idxes = bn.argpartition(-recon_batch[i], 10)[:10] # 유저에게 추천할 10개 영화를 가져옴
                tmp = list()
                # id2show 과정
                for j in idxes:                    
                    tmp.append(list(show2id.keys())[j]) # id2show # = tmp.append(raw_data['item'].unique()[j])
                submission_item.append(tmp)

    submission_item = np.array(submission_item).reshape(-1, 1)
    submission_user = raw_data['user'].unique().repeat(10)
    submission_user = np.array(submission_user).reshape(-1, 1)

    result = np.hstack((submission_user,submission_item))

    result = pd.DataFrame(result, columns=['user','item'])
    
    # XXX submission export 
    if args.data_process != 0: # 최근 데이터 일부만 사용한 경우
        # submission_data_최근 데이터 갯수.csv
        result.to_csv(os.path.join(args.output_dir, f'submission_data_{args.data_process}.csv'), index=False)
        print("export submission : ", os.path.join(args.output_dir, f'submission_data_{args.data_process}.csv'))
    elif args.data_random_process != 0:
        result.to_csv(os.path.join(args.output_dir, f'submission_data_random_{args.data_random_process}.csv'), index=False)
        print("export submission : ", os.path.join(args.output_dir, f'submission_data_random_{args.data_random_process}.csv'))
    elif args.train_all == True:
        result.to_csv(os.path.join(args.output_dir, f'submission_data_all_{args.batch_size}_100.csv'), index=False)
        print("export submission : ", os.path.join(args.output_dir, f'submission_data_all_{args.batch_size}_100.csv'))
    else : # 기본값             
        # submission_epoch 수_optimizer.csv
        result.to_csv(os.path.join(args.output_dir, f'submission_{args.epochs}_{args.optimizer}.csv'), index=False)
        print("export submission : ", os.path.join(args.output_dir, f'submission_{args.epochs}_{args.optimizer}.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", default="S3RecModel", type=str, help="model type") # model 모듈
    parser.add_argument("--model_name", default="MultiVAE", type=str)
    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)
    parser.add_argument("--output_dir", default="/opt/ml/input/code/output/", type=str)
    parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20,
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

    # XXX 추가한 args parse
    # dafault값은 모두 기존 코드와 동일한 기본값입니다.
    parser.add_argument("--wandb", type=bool, default=False, help="wandb") # wandb 사용 여부
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)') # optimizer 설정
    parser.add_argument('--data_process', type=int, default=0,  help='data process') # 최근 데이터를 얼마나 사용할 것인가
    parser.add_argument('--data_random_process', type=int, default=0,  help='data random process') # 데이터를 어느 비율만큼 랜덤으로 뽑을 것인가
    parser.add_argument('--train_all', type=bool, default=False,  help='use all training set') # 훈련데이터를 모두 쓸 것인지
    
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


    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"


    ###############################################################################
    # Load data
    ###############################################################################
    
    loader = DataLoader(args.data_dir)

    n_items = loader.load_n_items()
    train_data = loader.load_data('train')
    if args.train_all == False:
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

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        #weight_decay=args.wd
    )
    #criterion = loss_function_vae
    criterion = focal_loss

    ###############################################################################
    # Training code
    ###############################################################################

    #best_n100 = -np.inf
    best_r10 = -np.inf
    update_count = 0

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")

    # save model
    # XXX 저장 model 이름 변경
    # 여러 실험하기 위해 나누습니다
    
    if args.data_process != 0 : # 최근 데이터 일부만 사용하는 경우
        # 모델이름-데이터이름_data_최근 데이터 갯수
        checkpoint = f"{args_str}_data_{args.data_process}.pt"
    elif args.data_random_process != 0 : # 최근 데이터 일부만 사용하는 경우
        # 모델이름-데이터이름_data_random_비율
        checkpoint = f"{args_str}_data_random_{args.data_random_process}.pt"
    elif args.train_all == True:
        checkpoint = f"{args_str}_data_random_train_all_100.pt"
    else : # 기본값
        # 모델이름-데이터이름_epoch 수_optimizer.pt
        checkpoint = f"{args_str}_{args.epochs}_{args.optimizer}.pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)

    # XXX wandb 설정
    # 기본 값은 wandb를 사용하지 않습니다.
    if args.wandb:
        wandb.login()
        if args.data_process != 0 : # 최근 데이터만 뽑을 때
            wandb.init(group="Multi-VAE_data_process", project="MovieLens", entity="recsys-06", name=f"Multi-VAE_{args.data_process}_{args.optimizer}")
        elif args.data_random_process != 0: # 데이터를 랜덤으로 뽑을 때
            wandb.init(group="Multi-VAE_data_process", project="MovieLens", entity="recsys-06", name=f"Multi-VAE_random_{args.data_random_process}_{args.optimizer}")
        else : # 기본값
            wandb.init(group="Multi-VAE", project="MovieLens", entity="recsys-06", name=f"Multi-VAE_{args.epochs}_{args.optimizer}")
        
        wandb.config = args
    

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(model, criterion, optimizer, is_VAE=True) # 훈련
        if args.train_all == False:
            val_loss, n100, r10, r20, r50 = evaluate(model, criterion, vad_data_tr, vad_data_te, is_VAE=True) # 검증 데이터 테스트
        # XXX wnadb 설정
        if args.wandb:
            wandb.log({
                "val_loss" : val_loss,
                "n100" : n100,
                "r10" : r10,
                "r20" : r20,
                "r50" : r50
            })
        if args.train_all == False:
            print('-' * 100)
            print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                    'n100 {:5.3f} | r10 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                        epoch, time.time() - epoch_start_time, val_loss,
                        n100, r10, r20, r50))
            print('-' * 100)

            n_iter = epoch * len(range(0, N, args.batch_size))

        # Save the model if the n100 is the best we've seen so far.
        # with open(args.checkpoint_path, 'wb') as f:
        #     torch.save(model, f)
        # print(f"save model : {args.checkpoint_path}")
        if args.train_all == False:
            if r10 > best_r10:
                with open(args.checkpoint_path, 'wb') as f:
                    torch.save(model, f)
                best_r10 = r10
                print(f"save model : {args.checkpoint_path}")
        else:
            with open(args.checkpoint_path, 'wb') as f:
                torch.save(model, f)
            print(f"save model : {args.checkpoint_path}")

   
    if args.train_all == False:
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

    # Load the best saved model.
    with open(args.checkpoint_path, 'rb') as f:
        print(f"load model : {args.checkpoint_path}")
        model = torch.load(f)    

    # XXX submission 평가
    evaluate_submission(model, criterion=focal_loss, submission_data=submission_data, is_VAE=True)
