import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from models_sep import *


class ICMAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            eta=0.01,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            stack_size=1):
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.eta = eta
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.icm = ICMModel(input_size, output_size, use_cuda)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.icm.parameters()),
                                    lr=learning_rate)
        self.icm = self.icm.to(self.device)

        self.model = self.model.to(self.device)

        self.mdrnn = MDRNN(256, self.output_size, 256, 5).to(self.device)
        self.optimizer_rnn = optim.Adam(list(self.mdrnn.parameters()), lr=learning_rate)

    def get_action(self, state, prev_state, prev_action):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        prev_state = torch.Tensor(prev_state).to(self.device)
        prev_state = prev_state.float()
        prev_action = torch.LongTensor(prev_action).to(self.device)

        action_onehot = torch.FloatTensor(prev_action.shape[0], self.output_size).to(self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, prev_action.view(len(prev_action), -1), 1)
        action_onehot = action_onehot.reshape(1, self.num_env, self.output_size)

        policy, value = self.model(state, self.icm, self.mdrnn, prev_state, action_onehot)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(action_prob)

        return action, value.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, state, next_state, action):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        action_onehot = torch.FloatTensor(len(action), self.output_size).to(self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        real_next_state_feature = self.icm.features_forward(next_state)

        state_feature = self.icm.features_forward(state)

        action_onehot = action_onehot.reshape(1, self.num_env, self.output_size)
        state_feature = state_feature.reshape(1, self.num_env, state_feature.shape[1])
        pred_next_state_feature = self.mdrnn(action_onehot, state_feature)

        mus, sigmas, logpi, dones = pred_next_state_feature
        real = real_next_state_feature.reshape(1, self.num_env, real_next_state_feature.shape[1])

        intrinsic_reward = self.eta * gmm_loss(real, mus, sigmas, logpi, reduce=False) / real.shape[2]

        return intrinsic_reward.data.cpu().numpy()

    def train_model(self, s_batch, next_s_batch, prev_s_batch, prev_actions, target_batch, y_batch, adv_batch, old_policy, dones):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
        prev_s_batch = torch.FloatTensor(prev_s_batch).to(self.device)
        target_batch = torch.FloatTensor(target_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        prev_actions = torch.LongTensor(prev_actions).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        sample_range = np.arange(len(s_batch))

        ce = nn.CrossEntropyLoss()

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j: self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven
                action_onehot = torch.FloatTensor(self.batch_size, self.output_size).to(self.device)
                action_onehot.zero_()
                action_onehot.scatter_(1, y_batch[sample_idx].view(-1, 1), 1)
                pred_action = self.icm([s_batch[sample_idx], next_s_batch[sample_idx], action_onehot])

                inverse_loss = ce(
                    pred_action, y_batch[sample_idx])

                action_onehot = torch.FloatTensor(self.batch_size, self.output_size).to(self.device)
                action_onehot.zero_()
                action_onehot.scatter_(1, prev_actions[sample_idx].view(-1, 1), 1)

                action_onehot = action_onehot.reshape(1, -1, self.output_size)

                policy, value = self.model(s_batch[sample_idx], self.icm, self.mdrnn, prev_s_batch[sample_idx],\
                    action_onehot)
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(
                    value.sum(1), target_batch[sample_idx])

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                # loss = 0.1 * (actor_loss + 0.5 * critic_loss - 0.005 * entropy) + (0.2 * forward_loss + 0.8 * inverse_loss)
                loss = 0.1 * (actor_loss + 0.5 * critic_loss - 0.005 * entropy) + (0.8 * inverse_loss)
                loss.backward()

                self.optimizer.step()

            action_onehot = torch.FloatTensor(len(s_batch), self.output_size).to(self.device)
            action_onehot.zero_()
            action_onehot.scatter_(1, y_batch[:].view(-1, 1), 1)
            action_onehot = action_onehot.reshape(32, 64, self.output_size).transpose(0, 1)

            dones = dones.reshape(32, 64).transpose(0, 1)

            real_next_f_batch = self.icm.features_forward(next_s_batch)
            f_batch = self.icm.features_forward(s_batch)

            f_batch = f_batch.reshape(32, 64, f_batch.shape[1]).transpose(0, 1)
            real_next_f_batch = real_next_f_batch.reshape(32, 64, real_next_f_batch.shape[1]).transpose(0, 1)

            mus, sigmas, logpi, ds = self.mdrnn(action_onehot, f_batch.detach())

            self.optimizer_rnn.zero_grad()

            gmm = gmm_loss(real_next_f_batch.detach(), mus, sigmas, logpi)
            bce = F.binary_cross_entropy_with_logits(ds, dones)

            rnn_loss = ((gmm + bce) / (real_next_f_batch.shape[2] + 1)) * 0.2
            rnn_loss.backward()

            self.optimizer_rnn.step()

