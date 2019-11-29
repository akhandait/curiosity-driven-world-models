from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

from nes_py.wrappers import JoypadSpace
from tensorboardX import SummaryWriter

import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser("Evaluate")
parser.add_argument('--name', type=str, help="Name of the experiment to evaluate.")
parser.add_argument('--number', type=int, help="Model number.")
args = parser.parse_args()

# Name of the experiment to evaluate.
name = args.name
# Model number.
number = args.number

if 'shared' in name:
    from agents import *
else:
    from agents_sep import *

def main():
    print({section: dict(config[section]) for section in config.sections()})
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

    is_render = True
    model_path = 'models/{}/{}.model'.format(name, number)
    icm_path = 'models/{}/{}.icm'.format(name, number)
    mdrnn_path = 'models/{}/{}.mdrnn'.format(name, number)

    use_cuda = True
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = 1

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    stack_size = int(default_config['StateStackSize'])

    # eta
    eta = 0.25

    # sticky_action = False
    # action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

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
        use_noisy_net=use_noisy_net
    )

    print('Loading Pre-trained model....')
    if use_cuda:
        agent.model.load_state_dict(torch.load(model_path))
        agent.icm.load_state_dict(torch.load(icm_path))
        agent.mdrnn.load_state_dict(torch.load(mdrnn_path))
    # else:
        # agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        # agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
        # agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
    print('End load...')

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, life_done=life_done, history_size=stack_size)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, stack_size, 84, 84])
    prev_states = np.zeros([num_worker, stack_size, 84, 84])
    prev_actions = np.random.randint(0, output_size, size=(num_worker,))

    steps = 0
    rall = 0
    rd = False
    intrinsic_reward_list = []
    while not rd:
        steps += 1
        # actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)
        actions, _, policy = agent.get_action(np.float32(states) / 255.,
                                              np.float32(prev_states) / 255.,
                                              prev_actions)

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        # next_states, rewards, dones, real_dones, log_rewards, next_obs, max_x_pos = [], [], [], [], [], [], []
        for parent_conn in parent_conns:
            s, r, d, rd, lr, max_x_pos = parent_conn.recv()
            next_states = s.reshape([1, stack_size, 84, 84])

        prev_states = states
        states = next_states[:, :, :, :]
        prev_actions = actions


if __name__ == '__main__':
    main()
