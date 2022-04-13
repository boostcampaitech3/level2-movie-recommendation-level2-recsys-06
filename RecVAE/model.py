import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

def swish(x):
    return x*(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())

# class Sampling(nn.Module):
#     def __init__(self):
#         super(Sampling, self).__init__()
#
#     def forward(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = z_mean.shape[0]
#         dim = z_mean.shape[1]
#         epsilon = torch.normal(mean=torch.zeros(batch, dim), std=1.)
#         # epsilon = tf.keras.backend.random_normal(shape=(batch, dim), stddev=1.)
#         return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)


class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.decoder_resnet = nn.Linear(hidden_dim, input_dim)
        self.decoder_latent = nn.Linear(latent_dim, input_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # norm = x.pow(2).sum(dim=-1).sqrt()
        # x = x / norm[:, None]
        #
        # x = F.dropout(x, p=dropout_rate, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))

        dr = self.decoder_resnet(h5)
        dr = self.sig(dr)
        dl = self.decoder_latent(x)
        dl = self.sig(dl)

        return dr * dl

class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.ln6 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.ln7 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]

        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        h6 = self.ln5(swish(self.fc5(h5) + h1 + h2 + h3 + h4+ h5))
        h7 = self.ln5(swish(self.fc5(h6) + h1 + h2 + h3 + h4+ h5 + h6))
        return self.fc_mu(h7), self.fc_logvar(h7)
    

class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = Decoder(hidden_dim, latent_dim, input_dim)
        self.ease = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
        self.Sig = nn.Sigmoid()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        self.celoss = nn.CrossEntropyLoss()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def Sampling(self, z_mean_, z_log_var_):
        z_mean, z_log_var = z_mean_, z_log_var_
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.normal(mean=torch.zeros(batch, dim), std=1.)
        # epsilon = tf.keras.backend.random_normal(shape=(batch, dim), stddev=1.)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):

        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        EASE = self.ease(user_ratings).fill_diagonal_(0)

        z = self.reparameterize(mu, logvar)

        x_pred = self.decoder(z)
        if not calculate_loss:
            return x_pred
        EASE = self.Sig(EASE)
        # loss_result = self.kl_loss(user_ratings, x_pred * EASE)
        loss_result = self.celoss(user_ratings, x_pred*EASE)
        return x_pred * EASE, loss_result

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))