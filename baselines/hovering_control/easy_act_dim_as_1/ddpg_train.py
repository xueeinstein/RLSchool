#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import pickle
import parl
import argparse
import numpy as np
from threading import Thread
from rlschool import make_env
from parl.utils import logger, tensorboard, action_mapping, ReplayMemory

from quadrotor_agent import QuadrotorAgent
from quadrotor_model import QuadrotorModel

PARL_CLUSTER = '10.90.243.36:8003'


def load_hparams(hparams_file):
    with open(hparams_file, 'r') as f:
        hparams = json.load(f)
    return hparams


@parl.remote_class
class RemoteTrainer(object):
    def __init__(self, args, hparams):
        self.hparams = hparams

        if args.reward_type == 'R1':
            reward_func_hypers = [args.reward_type, args.alpha,
                                  args.beta, args.gamma]
        elif args.reward_type in ['R2', 'R3', 'R4']:
            reward_func_hypers = [args.reward_type, args.hovering_range,
                                  args.in_range_reward, args.out_range_reward]
        elif args.reward_type == 'R5':
            reward_func_hypers = [
                args.reward_type, args.alpha, args.hovering_range,
                args.in_range_reward, args.out_range_reward]
        self.env = make_env('Quadrotor',
                            task=args.task,
                            healthy_reward=args.healthy,
                            reward_func_hypers=reward_func_hypers)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        self.model = QuadrotorModel(act_dim)
        self.algorithm = parl.algorithms.DDPG(
            self.model,
            gamma=hparams['gamma'],
            tau=hparams['tau'],
            actor_lr=hparams['actor_lr'],
            critic_lr=hparams['critic_lr'])
        self.agent = QuadrotorAgent(self.algorithm, obs_dim, act_dim)

        self.rpm = ReplayMemory(hparams['memory_size'], obs_dim, act_dim)

    def run_evaluate_episode(self, log_steps):
        obs = self.env.reset()
        total_reward = 0
        steps = 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = self.agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)
            action = action_mapping(action, self.env.action_space.low[0],
                                    self.env.action_space.high[0])

            next_obs, reward, done, info = self.env.step(action)

            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break

        tb_logs = []
        tb_logs.append({
            'eval/episode_reward': total_reward,
            'log_steps': log_steps
        })
        tb_logs.append({
            'eval/episode_length': steps,
            'log_steps': log_steps
        })

        return tb_logs, total_reward, steps

    def run_train_episode(self, log_steps):
        obs = self.env.reset()
        total_reward = 0
        steps = 0
        tb_logs = []
        while True:
            steps += 1
            batch_obs = np.expand_dims(obs, axis=0)
            action = self.agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)

            # Add exploration noise, and clip to [-1.0, 1.0]
            action = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)
            action = action_mapping(action, self.env.action_space.low[0],
                                    self.env.action_space.high[0])

            next_obs, reward, done, info = self.env.step(action)

            self.rpm.append(
                obs, action, self.hparams['reward_scale'] * reward,
                next_obs, done)

            if self.rpm.size() > self.hparams['rpm_init_size']:
                batch_obs, batch_action, batch_reward, batch_next_obs, \
                    batch_terminal = self.rpm.sample_batch(
                        self.hparams['batch_size'])
                critic_cost = self.agent.learn(
                    batch_obs, batch_action, batch_reward,
                    batch_next_obs, batch_terminal)
                tb_logs.append({
                    'train/critic_loss': critic_cost,
                    'log_steps': log_steps + steps
                })

            obs = next_obs
            total_reward += reward

            if done:
                break

        tb_logs.append({
            'train/episode_reward': total_reward,
            'log_steps': log_steps + steps
        })
        return tb_logs, total_reward, steps

    def get_weights(self):
        return self.model.get_weights()


def log_tensorbord(idx, tb_logs):
    for tb_log in tb_logs:
        k1, k2 = list(tb_log.keys())
        steps = tb_log['log_steps']
        name = k1 if k1 != 'log_steps' else k2
        value = tb_log[name]
        tensorboard.add_scalar(name + '_{}'.format(idx), value, steps)


def run(idx, args, hparams):
    hparams_name = os.path.basename(args.hparams_file)[:-len('.json')]
    if args.reward_type == 'R1':
        unique_dir = '{}_alpha_{}_beta_{}_gamma_{}_hr_{}_{}'.format(
            args.reward_type, args.alpha, args.beta, args.gamma,
            args.healthy, hparams_name)
    elif args.reward_type in ['R2', 'R3', 'R4']:
        unique_dir = '{}_range_{}_in_{}_out_{}_hr_{}_{}'.format(
            args.reward_type, args.hovering_range, args.in_range_reward,
            args.out_range_reward, args.healthy, hparams_name)
    elif args.reward_type == 'R5':
        unique_dir = '{}_alpha_{}_range_{}_in_{}_out_{}_hr_{}_{}'.format(
            args.reward_type, args.alpha, args.hovering_range,
            args.in_range_reward, args.out_range_reward,
            args.healthy, hparams_name)
    else:
        unique_dir = 'test'

    model_dir = os.path.join(
        args.model_dir, unique_dir, 'run_{}'.format(idx))
    logdir = os.path.join(args.logdir, unique_dir, 'run_{}'.format(idx))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logger.set_dir(logdir)
    parl.connect(PARL_CLUSTER)
    trainer = RemoteTrainer(args, hparams)

    test_flag = 0
    total_steps = 0
    while total_steps < args.train_total_steps:
        tb_logs, train_reward, steps = trainer.run_train_episode(total_steps)
        total_steps += steps
        logger.info('Run: {}, Steps: {} Reward: {}'.format(
            idx, total_steps, train_reward))
        log_tensorbord(idx, tb_logs)

        if total_steps // args.test_every_steps >= test_flag:
            while total_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            tb_logs, evaluate_reward, steps = trainer.run_evaluate_episode(
                total_steps)
            msg = 'Run: {}, Steps {}, Eval reward: {}, Episode length: {}'
            logger.info(msg.format(idx, total_steps, evaluate_reward, steps))
            log_tensorbord(idx, tb_logs)

            ckpt = 'steps_{}.ckpt'.format(total_steps)
            with open(os.path.join(model_dir, ckpt), 'wb') as f:
                pickle.dump(trainer.get_weights(), f)


def main(args):
    hparams = load_hparams(args.hparams_file)

    thread_pool = [None] * args.parallel_run
    for i in range(args.parallel_run):
        thread_pool[i] = Thread(
            target=run, args=(i, args, hparams))
        thread_pool[i].start()

    for thread in thread_pool:
        thread.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train DDGP baseline model for Quadrotor environment')
    parser.add_argument(
        '--task', type=str, default='hovering_control',
        help='The default task name.')
    parser.add_argument(
        '--parallel-run', type=int, default=5,
        help='Number of parallel running.')
    parser.add_argument(
        '--train_total_steps', type=int, default=int(1e7),
        help='Maximum training steps.')
    parser.add_argument(
        '--test_every_steps', type=int, default=int(1e4),
        help='The step interval between two consecutive evaluations.')
    parser.add_argument(
        '--hparams_file', type=str, default='hparams.json',
        help='The path to hyperparameters configuration file.')
    parser.add_argument(
        '--logdir', type=str, default='logdir',
        help='The directory to training log.')
    parser.add_argument(
        '--model_dir', type=str, default='saved_models',
        help='The directory to save model parameters.')
    parser.add_argument(
        '--reward-type', type=str, default='R1',
        help='The reward function type: R1, R2 or R3.')
    parser.add_argument(
        '--healthy', type=float, default=10.0,
        help='Healthy reward for hovering control task.')

    R1 = parser.add_argument_group('R1')
    R1.add_argument(
        '--alpha', type=float, default=1.0,
        help='The alpha coefficient of reward function for hovering control')
    R1.add_argument(
        '--beta', type=float, default=1.0,
        help='The beta coefficient of reward function for hovering control')
    R1.add_argument(
        '--gamma', type=float, default=1.0,
        help='The gamma coefficient of reward function for hovering control')

    R2 = parser.add_argument_group('R2_or_R3_or_R4')
    R2.add_argument(
        '--hovering-range', type=float, default=0.5,
        help='Tolerant range of hovering.')
    R2.add_argument(
        '--in-range-reward', type=float, default=10,
        help='Reward when quadrotor is in hovering range.')
    R2.add_argument(
        '--out-range-reward', type=float, default=-20,
        help='Reward when quadrotor is out of hovering range.')
    args = parser.parse_args()

    main(args)
