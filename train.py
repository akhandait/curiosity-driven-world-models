from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

from nes_py.wrappers import JoypadSpace
from tensorboardX import SummaryWriter

import numpy as np
import copy
import os
import pickle
import argparse

parser = argparse.ArgumentParser("Evaluate")
parser.add_argument('--shared_features', action='store_true', help="")
args = parser.parse_args()

if args.shared_features:
    from agents import *
else:
    from agents_sep import *

def main():
    name = 'submission'
    print(name)
    try:
        os.makedirs('models/' + name)
    except OSError:
        pass

    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        env = JoypadSpace(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_load_model = False
    # Render
    is_render = False

    model_path = 'models/{}.model'.format(env_id)
    icm_path = 'models/{}.icm'.format(env_id)

    writer = SummaryWriter('runs/' + name)

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])

    eta = float(default_config['ETA'])

    stack_size = int(default_config['StateStackSize'])

    clip_grad_norm = float(default_config['ClipGradNorm'])

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, stack_size, 84, 84))

    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(gamma)

    agent = ICMAgent

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    else:
        raise NotImplementedError

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        eta=eta,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        stack_size=stack_size
    )

    if is_load_model:
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
            agent.icm.load_state_dict(torch.load(icm_path))
            agent.mdrnn.load_state_dict(torch.load(mdrnn_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, history_size=stack_size)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, stack_size, 84, 84])
    prev_states = np.zeros([num_worker, stack_size, 84, 84])
    prev_actions = np.random.randint(0, output_size, size=(num_worker,))

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # normalize obs
    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    steps = 0
    while steps < pre_obs_norm_step:
        steps += num_worker
        actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd, lr, max_x = parent_conn.recv()
            next_obs.append(s[:])

    next_obs = np.stack(next_obs)
    obs_rms.update(next_obs)
    print('End to initalize...')

    rewards_list = []
    intrinsic_reward_list = []
    max_x_pos_list = []
    samples_ep_list = []
    global_update_list = []
    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_prev_state, total_prev_action, \
            total_int_reward, total_next_obs, total_values, total_policy, total_log_reward = \
            [], [], [], [], [], [], [], [], [], [], [], []

        global_step += (num_worker * num_step)
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(num_step):
            actions, value, policy = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var),
                                                      (prev_states - obs_rms.mean) / np.sqrt(obs_rms.var),
                                                      prev_actions)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards, next_obs, max_x_pos = [], [], [], [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd, lr, max_x = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                max_x_pos.append(max_x)

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            log_rewards = np.hstack(log_rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)

            # print(rewards.shape)
            # total reward = int reward
            # print(states.shape)
            intrinsic_reward = agent.compute_intrinsic_reward(
                (states - obs_rms.mean) / np.sqrt(obs_rms.var),
                (next_states - obs_rms.mean) / np.sqrt(obs_rms.var),
                actions).reshape(16,)
            # print(intrinsic_reward.shape)

            sample_i_rall += intrinsic_reward[sample_env_idx]
            # sample_i_rall += intrinsic_reward
            # print(intrinsic_reward)
            # print(intrinsic_reward)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_next_state.append(next_states)
            total_prev_state.append(prev_states)
            total_prev_action.append(prev_actions)
            total_reward.append(rewards)
            total_log_reward.append(log_rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_values.append(value)
            total_policy.append(policy)
            # print(len(total_reward))

            # Edit.
            prev_states = states
            states = next_states[:, :, :, :]
            prev_actions = actions

            sample_rall += log_rewards[sample_env_idx]
            # print(sample_env_idx)
            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                writer.add_scalar('data/step', sample_step, sample_episode)
                writer.add_scalar('data/int_reward_per_epi', sample_i_rall, sample_episode)
                writer.add_scalar('data/int_reward_per_rollout', sample_i_rall, global_update)
                writer.add_scalar('data/max_x_pos_per_epi', max_x_pos[sample_env_idx], sample_episode)

                rewards_list.append(sample_rall)
                intrinsic_reward_list.append(sample_i_rall)
                max_x_pos_list.append(max_x_pos[sample_env_idx])
                samples_ep_list.append(sample_episode)
                global_update_list.append(global_update)

                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0


        # calculate last next value
        _, value, _ = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var),
                                       (prev_states - obs_rms.mean) / np.sqrt(obs_rms.var),
                                        prev_actions)
        total_values.append(value)
        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, stack_size, 84, 84])
        total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, stack_size, 84, 84])
        total_prev_state = np.stack(total_prev_state).transpose([1, 0, 2, 3, 4]).reshape([-1, stack_size, 84, 84])
        total_prev_action = np.stack(total_prev_action).transpose().reshape([-1])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose().reshape([-1])
        total_values = np.stack(total_values).transpose()
        total_logging_policy = torch.stack(total_policy).view(-1, output_size).cpu().numpy()

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward = np.stack(total_reward).transpose()
        total_log_reward = np.stack(total_log_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T + total_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        total_reward /= np.sqrt(reward_rms.var)

        writer.add_scalar('data/normalized_int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/normalized_int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        writer.add_scalar('data/total_reward_per_epi', np.sum(total_reward) / num_worker, sample_episode)
        writer.add_scalar('data/total_reward_per_rollout', np.sum(total_reward) / num_worker, global_update)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # print(total_reward.shape)
        total_int_reward = total_int_reward.reshape(num_worker, num_step)
        total_reward = total_reward.reshape(num_worker, num_step)
        # Step 3. make target and advantage
        target, adv = make_train_data(total_int_reward + total_reward,
                                      np.zeros_like(total_int_reward),
                                      total_values.reshape(num_worker, num_step + 1),
                                      gamma,
                                      num_step,
                                      num_worker)

        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        # -----------------------------------------------

        agent.train_model((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                          (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                          (total_prev_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                          total_prev_action,
                          target, total_action,
                          adv,
                          total_policy, total_done)

        if global_step % (num_worker * num_step * 10) == 0:
            with open('losses/' + name + '.pkl', 'wb') as f:
                pickle.dump((rewards_list, intrinsic_reward_list, max_x_pos_list, samples_ep_list, \
                    global_update_list), f)

        if global_step % (num_worker * num_step * 100) == 0:
            print('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), 'models/' + name + '/' + str(global_step) + '.model')
            torch.save(agent.icm.state_dict(), 'models/' + name + '/' + str(global_step) + '.icm')
            torch.save(agent.mdrnn.state_dict(), 'models/' + name + '/' + str(global_step) + '.mdrnn')

if __name__ == '__main__':
    main()
