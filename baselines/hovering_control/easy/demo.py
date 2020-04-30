import argparse
import numpy as np
from paddle import fluid
from parl import layers
from parl.utils import action_mapping
from rlschool import make_env

from quadrotor_model import QuadrotorModel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize the agent performance.')
    parser.add_argument(
        '--ckpt', type=str, default=None,
        help='The path of model checkpoint file.')
    parser.add_argument(
        '--task', type=str, default='hovering_control',
        help='The default task name.')
    return parser.parse_args()


def main(args):
    env = make_env('Quadrotor', task=args.task)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if args.task == 'velocity_control':
        obs_dim += 3

    model = QuadrotorModel(act_dim)
    pred_program, startup_program = fluid.Program(), fluid.Program()
    with fluid.program_guard(pred_program, startup_program):
        obs = layers.data(
            name='obs', shape=[obs_dim], dtype='float32')
        pred_act = model.policy(obs)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)
    model.load(args.ckpt, actor_only=True)

    def predict(obs):
        obs = np.expand_dims(obs, axis=0)
        act = exe.run(
            pred_program, feed={'obs': obs}, fetch_list=[pred_act])[0]
        return act

    obs = env.reset()
    env.render()

    steps, total_reward = 0, 0
    while True:
        act = predict(obs.astype('float32'))
        act = np.squeeze(act)
        act = action_mapping(act, 0.1, 15.0)

        next_obs, reward, done, info = env.step(act)
        print(act, reward)
        env.render()

        obs = next_obs
        total_reward += reward
        steps += 1

        if done:
            break

    print('Episode steps: {}, Total reward: {}'.format(steps, total_reward))


if __name__ == '__main__':
    args = parse_args()
    assert args.ckpt is not None
    main(args)
