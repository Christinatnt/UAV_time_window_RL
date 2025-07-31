import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
from scipy import stats
import os
from datetime import datetime
from config import *

# 设置中文字体支持
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial"]
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300  # 提高图像分辨率


class Plotter:
    def __init__(self, save_dir='results/plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def smooth_curve(self, y, window=10):
        if len(y) < window:
            return y
        box = np.ones(window) / window
        y_smooth = np.convolve(y, box, mode='same')
        for i in range(window // 2):
            y_smooth[i] = np.mean(y[:i + window // 2 + 1])
            y_smooth[-(i + 1)] = np.mean(y[-(i + window // 2 + 1):])
        return y_smooth

    def peak_preserving_smooth(self, y, window=5):
        """保持峰值不变的平滑方法"""
        # 1. 先识别原始极值点
        peaks = np.where((y > np.roll(y, 1)) )[0]
        # peaks = np.where((y > np.roll(y, 1)) & (y > np.roll(y, -1)))[0]

        # 2. 常规平滑
        box = np.ones(window) / window
        y_smooth = np.convolve(y, box, mode='same')

        # 3. 将极值点恢复为原始值
        y_smooth[peaks] = y[peaks]  # 关键步骤

        # 4. 边界处理
        for i in range(window // 2):
            y_smooth[i] = y[i]
            y_smooth[-(i + 1)] = y[-(i + 1)]

        return y_smooth

    def plot_rewards(self, rewards, title,
                     xlabel, ylabel,
                     smooth=True, window=10,
                     save_name=None):
        plt.figure(figsize=(10, 6))
        rewards = np.array(rewards)
        mean_rewards = np.mean(rewards, axis=0)
        x = np.arange(1, len(mean_rewards) + 1)

        if smooth:
            mean_smooth = self.smooth_curve(mean_rewards, window)
            plt.plot(x, mean_smooth, 'b-', linewidth=2, label='平均奖励')
        else:
            plt.plot(x, mean_rewards, 'b-', linewidth=2, label='平均奖励')

        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.legend(fontsize=12)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        y_min = min(np.min(mean_rewards), 0)
        plt.ylim(y_min, plt.ylim()[1])

        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), bbox_inches='tight')
        return plt

    def plot_multiple_algorithms(self, rewards_dict, title,
                                 xlabel, ylabel,
                                 smooth=True, window=10,
                                 save_name=None):
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10.colors

        for i, (algo_name, rewards) in enumerate(rewards_dict.items()):
            rewards = np.array(rewards)
            mean_rewards = np.mean(rewards, axis=0)
            x = np.arange(1, len(mean_rewards) + 1)

            if smooth:
                mean_smooth = self.smooth_curve(mean_rewards, window)
                plt.plot(x, mean_smooth, color=colors[i], linewidth=2, label=algo_name)
            else:
                plt.plot(x, mean_rewards, color=colors[i], linewidth=2, label=algo_name)

        # plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        y_min = min(min(np.min(np.mean(np.array(r), axis=0)) for r in rewards_dict.values()), 0)
        plt.ylim(y_min, plt.ylim()[1])

        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), bbox_inches='tight')
        # return plt

    # 绘制不同时间窗进度对比折线图
    def plot_multiple_reward_components(self, rewards_dict, title,
                                        xlabel, ylabel,
                                        smooth=True, window=10,
                                        save_name=None):
        """
        :param reward_histories: List of dict，每个dict对应一种时间窗设置，包含各分量曲线
        :param labels: 与 reward_histories 一一对应的标签
        """
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10.colors
        for i, (algo_name, rewards) in enumerate(rewards_dict.items()):
            rewards = np.array(rewards)
            mean_rewards = np.mean(rewards, axis=0)
            x = np.arange(1, len(mean_rewards) + 1)

            if smooth:
                mean_smooth = self.smooth_curve(mean_rewards, window)
                plt.plot(x, mean_smooth, color=colors[i], linewidth=2, label=algo_name)
            else:
                plt.plot(x, mean_rewards, color=colors[i], linewidth=2, label=algo_name)

        # plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        y_min = min(min(np.min(np.mean(np.array(r), axis=0)) for r in rewards_dict.values()), 0)
        plt.ylim(y_min, plt.ylim()[1])

        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), bbox_inches='tight')
        plt.show()
        # return plt

    def plot_uav_trajectories(self, uav_trajectories, task_positions, title, save_name=""):
        """
        uav_trajectories: List of [num_steps x 2] for each UAV
        task_positions: List or ndarray of task positions [num_tasks x 2]
        """
        plt.figure(figsize=(8, 8))

        # 绘制任务点
        task_positions = np.array(task_positions)
        plt.scatter(task_positions[:, 0], task_positions[:, 1], c='red', marker='x', label='task', s=30)

        # 绘制 UAV 轨迹
        for i, traj in enumerate(uav_trajectories):
            traj = np.array(traj)
            plt.scatter(traj[0, 0], traj[0, 1], c='green', marker='o', s=30,
                        label=f'UAV starting point' if i == 0 else None)
            plt.plot(traj[:, 0], traj[:, 1], label=f'UAV {i}')

        plt.title(title, fontsize=14)
        plt.xlabel("X-axis (m)", fontsize=12)
        plt.ylabel("Y-axis (m)", fontsize=12)
        plt.xlim(0, MAP_SIZE)
        plt.ylim(0, MAP_SIZE)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.grid(True)
        # plt.axis("equal")

        if save_name:
            # plt.savefig(os.path.join(self.save_dir, f"{save_name}.pdf"), bbox_inches='tight')
            plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), bbox_inches='tight')
        # return plt
        plt.show()

    # 绘制GA奖励函数曲线
    def plot_reward_components(self, reward_history):
        for key, values in reward_history.items():
            values = np.array(values)
            if values.ndim == 2:
                avg_values = np.mean(values, axis=1)  # 每个时间步的平均奖励
            else:
                avg_values = values  # 如果是一维的就直接绘图
            plt.plot(avg_values, label=key)
        plt.title("各时间步平均奖励分量")
        plt.xlabel("时间步")
        plt.ylabel("奖励值")
        plt.legend()
        plt.grid(True)
        # plt.savefig(os.path.join(self.save_dir, f"reward_compare.png"), bbox_inches='tight')
        plt.show()
