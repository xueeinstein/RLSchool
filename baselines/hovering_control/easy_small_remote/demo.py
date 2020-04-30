import pickle
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

    R2 = parser.add_argument_group('R2_or_R3')
    R2.add_argument(
        '--hovering-range', type=float, default=0.5,
        help='Tolerant range of hovering.')
    R2.add_argument(
        '--in-range-reward', type=float, default=10,
        help='Reward when quadrotor is in hovering range.')
    R2.add_argument(
        '--out-range-reward', type=float, default=-20,
        help='Reward when quadrotor is out of hovering range.')
    return parser.parse_args()


def main(args):
    if args.reward_type == 'R1':
        reward_func_hypers = [args.reward_type, args.alpha,
                              args.beta, args.gamma]
    elif args.reward_type in ['R2', 'R3', 'R4', 'R7']:
        reward_func_hypers = [args.reward_type, args.hovering_range,
                              args.in_range_reward, args.out_range_reward]
    elif args.reward_type == 'R5':
        reward_func_hypers = [
            args.reward_type, args.alpha, args.hovering_range,
            args.in_range_reward, args.out_range_reward]
    elif args.reward_type in ['R6', 'R8']:
        reward_func_hypers = [
            args.reward_type, args.alpha, args.beta,
            args.hovering_range, args.in_range_reward,
            args.out_range_reward]

    env = make_env('Quadrotor',
                   task=args.task,
                   healthy_reward=args.healthy,
                   reward_func_hypers=reward_func_hypers)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if args.task == 'velocity_control':
        obs_dim += 3

    with open(args.ckpt, 'rb') as f:
        ckpt = pickle.load(f)
    model = QuadrotorModel(act_dim)
    pred_program, startup_program = fluid.Program(), fluid.Program()
    with fluid.program_guard(pred_program, startup_program):
        obs = layers.data(
            name='obs', shape=[obs_dim], dtype='float32')
        pred_act = model.policy(obs)

    # @parl-dev: when use `model.get_weights()', we must create the *full*
    # network, otherwise cannot use `model.set_weights' to init the weights
    dummy_program, dummy_startup_program = fluid.Program(), fluid.Program()
    with fluid.program_guard(dummy_program, dummy_startup_program):
        obs = layers.data(
            name='obs', shape=[obs_dim], dtype='float32')
        act = layers.data(
            name='act', shape=[act_dim], dtype='float32')
        model.value(obs, act)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)
    exe.run(dummy_startup_program)
    model.set_weights(ckpt)

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
        print(info)
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
