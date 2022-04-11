@@ -0,0 +1,234 @@
import numpy as np

import torch
from torch import optim

import random
from copy import deepcopy

from utils import get_data, recall
from model import VAE

import pandas as pd
import bottleneck as bn

import wandb
wandb.init(
        project="MovieLens", 
        entity="recsys-06",  
        name="RecVAE beta 0.4",
        notes="recall 10",
        group="RecVAE"
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--hidden-dim', type=int, default=600)
parser.add_argument('--latent-dim', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=500)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n-epochs', type=int, default=50)
parser.add_argument('--n-enc_epochs', type=int, default=3)
parser.add_argument('--n-dec_epochs', type=int, default=1)
parser.add_argument('--not-alternating', type=bool, default=False)
args = parser.parse_args()

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0")

data = get_data(args.dataset)
# train_data, test_in_data, test_out_data = data
train_data, = data # 데이터 전체로 학습 후 결과하기 위해


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
        for batch in generate(batch_size=batch_size, device=device, data_in=train_data, shuffle=True):
            ratings = batch.get_ratings_to_dev()

            for optimizer in opts:
                optimizer.zero_grad()

            _, loss = model(ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
            loss.backward()

            for optimizer in opts:
                optimizer.step()


model_kwargs = {
    'hidden_dim': args.hidden_dim,
    'latent_dim': args.latent_dim,
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
    'batch_size': args.batch_size,
    'beta': args.beta,
    'gamma': args.gamma
}

decoder_params = set(model.decoder.parameters())
encoder_params = set(model.encoder.parameters())

optimizer_encoder = optim.Adam(encoder_params, lr=args.lr)
optimizer_decoder = optim.Adam(decoder_params, lr=args.lr)


for epoch in range(args.n_epochs):

    if args.not_alternating:
        run(opts=[optimizer_encoder, optimizer_decoder], n_epochs=1, dropout_rate=0.5, **learning_kwargs)
    else:
        run(opts=[optimizer_encoder], n_epochs=args.n_enc_epochs, dropout_rate=0.5, **learning_kwargs)
        model.update_prior()
        run(opts=[optimizer_decoder], n_epochs=args.n_dec_epochs, dropout_rate=0, **learning_kwargs)

    train_scores.append(
        evaluate(model, train_data, train_data, metrics, 0.01)[0]
    )

    wandb.log({'score': train_scores[-1]})

    if train_scores[-1] > best_recall:
        best_recall = train_scores[-1]
        model_best.load_state_dict(deepcopy(model.state_dict()))


    print(f'epoch {epoch} | train recall@10: {train_scores[-1]:.4f}')



# test_metrics =  [{'metric': recall, 'k': 10}]

# final_scores = evaluate(model_best, test_in_data, test_out_data, test_metrics)

# for metric, score in zip(test_metrics, final_scores):
#     print(f"{metric['metric'].__name__}@{metric['k']}:\t{score:.4f}")

# torch.save(model_best.state_dict(), './RecVAE epochs 60 beta 0.4 latent-dim 250.pth')

def result(model, data_in, samples_perc_per_epoch=1, batch_size=500):
    model.eval()
    items=[]
    user = pd.read_csv('../unique_uid.csv', header=None)
    item = pd.read_csv('../unique_sid.csv', header=None)
    item = item.to_numpy()
    for batch in generate(batch_size=batch_size,
                          device=device,
                          data_in=data_in,
                          samples_perc_per_epoch=samples_perc_per_epoch
                         ):

        ratings_in = batch.get_ratings_to_dev()

        ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()

        ratings_pred[batch.get_ratings().nonzero()] = -np.inf


        batch_users = ratings_pred.shape[0]
        idx = bn.argpartition(-ratings_pred, 10, axis=1)
        X_pred_binary = np.zeros_like(ratings_pred, dtype=bool)
        X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :10]] = True
        for i in X_pred_binary:
            items.append(item[i])
    users = np.array(user)
    users = users.repeat(10).reshape(-1,1)
    items = np.array(items).reshape(-1,1)
    result = np.concatenate((users,items),axis=1)
    result = pd.DataFrame(result, columns=['user','item'])
    result.to_csv('result.csv', index=False)

# result(model_best,train_data) 
