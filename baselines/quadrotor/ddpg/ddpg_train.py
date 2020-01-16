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
import parl
import argparse
import numpy as np
from rlschool import make_env
from parl.utils import logger, tensorboard, action_mapping, ReplayMemory

from quadrotor_agent import QuadrotorAgent
from quadrotor_model import QuadrotorModel


def load_hparams(hparams_file):
    with open(hparams_file, 'r') as f:
        hparams = json.load(f)
    return hparams


def run_train_episode(env, agent, rpm, hparams, log_steps):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)

        # Add exploration noise, and clip to [-1.0, 1.0]
        action = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        # next_obs, reward, done, info = env.step(action)
        next_obs, reward, done = env.step(action)

        rpm.append(
            obs, action, hparams['reward_scale'] * reward, next_obs, done)

        if rpm.size() > hparams['rpm_init_size']:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
                batch_terminal = rpm.sample_batch(hparams['batch_size'])
            critic_cost = agent.learn(
                batch_obs, batch_action, batch_reward,
                batch_next_obs, batch_terminal)
            tensorboard.add_scalar(
                'train/critic_loss', critic_cost, log_steps + steps)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


def run_evaluate_episode(env, agent, log_steps):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        # next_obs, reward, done, info = env.step(action)
        next_obs, reward, done = env.step(action)

        obs = next_obs
        total_reward += reward
        steps += 1

        if done:
            break

    logger.info('Evaluate episode length: {}'.format(steps))
    tensorboard.add_scalar('eval/episode_length', steps, log_steps)
    return total_reward


def main(args):
    if args.logdir is None:
        logger.auto_set_dir()
    else:
        logger.set_dir(args.logdir)

    hparams = load_hparams(args.hparams_file)
    env = make_env('Quadrotor', task=args.task)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = QuadrotorModel(act_dim)
    algorithm = parl.algorithms.DDPG(
        model,
        gamma=hparams['gamma'],
        tau=hparams['tau'],
        actor_lr=hparams['actor_lr'],
        critic_lr=hparams['critic_lr'])
    agent = QuadrotorAgent(algorithm, obs_dim, act_dim)

    rpm = ReplayMemory(hparams['memory_size'], obs_dim, act_dim)

    test_flag = 0
    total_steps = 0
    while total_steps < args.train_total_steps:
        train_reward, steps = run_train_episode(
            env, agent, rpm, hparams, total_steps)
        total_steps += steps
        logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward))
        tensorboard.add_scalar(
            'train/episode_reward', train_reward, total_steps)

        if total_steps // args.test_every_steps >= test_flag:
            while total_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            evaluate_reward = run_evaluate_episode(env, agent, total_steps)
            logger.info('Steps {}, Evaluate reward: {}'.format(
                total_steps, evaluate_reward))
            tensorboard.add_scalar(
                'eval/episode_reward', evaluate_reward, total_steps)

            ckpt = 'steps_{}.ckpt'.format(total_steps)
            model.save(os.path.join(args.model_dir, ckpt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train DDGP baseline model for Quadrotor environment')
    parser.add_argument(
        '--task', type=str, default='no_collision',
        help='The default task name.')
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
        '--logdir', type=str, default=None,
        help='The directory to training log.')
    parser.add_argument(
        '--model_dir', type=str, default='save',
        help='The directory to save model parameters.')
    args = parser.parse_args()

    main(args)
