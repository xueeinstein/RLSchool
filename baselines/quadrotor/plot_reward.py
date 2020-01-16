import glob
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot reward curve from training log.')
    parser.add_argument(
        '--csv-pattern', type=str, default=None,
        help='File path pattern to get multiple runs log.'
        'In format "algo1:algo1_run*.csv,algo2:algo2_run*.csv".')
    parser.add_argument(
        '--max-steps', type=int, default=int(1e6),
        help='Max steps to plot. Omit reward larger than max steps.')
    parser.add_argument(
        '--interval-steps', type=int, default=int(1e4),
        help='Interval steps of reward curve.')
    return parser.parse_args()


def get_reward_curve_data(csv, max_steps, interval_steps):
    df = pd.read_csv(csv)
    rewards = []
    for i in range(len(df.Step)):
        if df.Step[i] // interval_steps < max_steps // interval_steps + 1:
            rewards.append(df.Value[i])

    print('max:', max(rewards))
    return rewards


def plot_reward_curve(args):
    algos_lst = [i.split(':') for i in args.csv_pattern.split(',')]

    algo_rewards = dict()
    for algo, runs in algos_lst:
        algo_rewards[algo] = []
        for csv in glob.glob(runs):
            rewards = get_reward_curve_data(
                csv, args.max_steps, args.interval_steps)
            algo_rewards[algo].append(rewards)

    xdata = list(range(0, args.max_steps + 1, args.interval_steps))

    colors = ['r', 'g', 'b']
    assert len(colors) >= len(algo_rewards)

    ax_lst = []
    plt.figure()
    for i, algo in enumerate(algo_rewards.keys()):
        ax = sns.tsplot(time=xdata, data=algo_rewards[algo], color=colors[i])
        ax_lst.append(ax)

    plt.legend(ax_lst, labels=list(algo_rewards.keys()))

    plt.show()


if __name__ == '__main__':
    args = parse_args()
    plot_reward_curve(args)
