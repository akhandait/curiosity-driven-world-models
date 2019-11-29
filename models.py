import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from torch.distributions.normal import Normal

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False):
        super(CnnActorCriticNetwork, self).__init__()

        # if use_noisy_net:
        #     print('use NoisyNet')
        #     linear = NoisyLinear
        # else:
        linear = nn.Linear

        # self.feature = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=1,
        #         out_channels=32,
        #         kernel_size=8,
        #         stride=4),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(
        #         in_channels=32,
        #         out_channels=64,
        #         kernel_size=4,
        #         stride=2),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(
        #         in_channels=64,
        #         out_channels=64,
        #         kernel_size=3,
        #         stride=1),
        #     nn.LeakyReLU(),
        #     Flatten(),
        #     linear(
        #         7 * 7 * 64,
        #         512),
        #     nn.LeakyReLU()
        # )

        self.actor = nn.Sequential(
            linear(512, 512),
            nn.LeakyReLU(),
            linear(512, output_size)
        )

        self.critic = nn.Sequential(
            linear(512, 512),
            nn.LeakyReLU(),
            linear(512, 1)
        )

        # self.actor = nn.Sequential(
        #     linear(512, output_size)
        # )

        # self.critic = nn.Sequential(
        #     linear(512, 1)
        # )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.critic)):
            if type(self.critic[i]) == nn.Linear:
                init.orthogonal_(self.critic[i].weight, 0.01)
                self.critic[i].bias.data.zero_()

    def forward(self, state, icm, rnn, prev_state, prev_action):
        # x = self.feature(state)

        feat_prev_s = icm.features_forward(prev_state).reshape(1, prev_state.shape[0], 256)
        rnn_h = F.leaky_relu(rnn.get_hidden(prev_action, feat_prev_s).detach()).reshape(feat_prev_s.shape[1], 256)
        feat_s = icm.features_forward(state).detach()

        x = torch.cat([rnn_h, feat_s], dim=-1)

        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class ICMModel(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=True):
        super(ICMModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(7 * 7 * 64, 256)
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def features_forward(self, state):
        return F.leaky_relu(self.feature(state))

    def forward(self, inputs):
        state, next_state, action = inputs

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)

        return pred_action


class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, *inputs):
        pass

class MDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

        for name, param in self.rnn.named_parameters():
            init.uniform_(param)
            if 'weight' in name:
                init.kaiming_uniform_(param, a=1.0)
            else:
                param.data.zero_()

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor # NOTE: not needed, use batch_first in LSTM
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = F.log_softmax(pi, dim=-1)

        dones = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, dones

    def get_hidden(self, actions, latents):
        ins = torch.cat([actions, latents], dim=-1)
        outs, (h, c) = self.rnn(ins)

        return h

def gmm_loss(batch, mus, sigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
    """ Computes the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob
