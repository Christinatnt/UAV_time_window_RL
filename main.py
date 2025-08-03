import time
import sys
import os
from maddpg import MADDPG
from config import TIME_STEPS
from Plotter import Plotter
import numpy as np
import random
from datetime import datetime
import tensorflow as tf
from GA import GA
from env import UAVEnv
from MAPPO import MAPPO
import json


def run_with_MADDPG(plotter, time_window_type, epochs=500):
    env = UAVEnv(time_window_type)
    maddpg = MADDPG(env)
    all_reward, all_completion, all_collision, all_fairness = maddpg.train(epochs)
    # 绘制奖励曲线
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 奖励曲线
    plotter.plot_rewards([all_reward],
                         title="UAV Reward Curve",
                         xlabel="Iteration number",
                         ylabel="Reward",
                         smooth=True,
                         window=5,
                         # save_name=f"MADDPG_reward_curve_{timestamp}")
                         save_name=f"MADDPG_reward_curve")

    # 任务完成率曲线
    plotter.plot_rewards([all_completion],
                         title="UAV Task Completion Rate",
                         xlabel="Iteration number",
                         ylabel="Task completion rate",
                         smooth=True,
                         window=5,
                         # save_name=f"MADDPG_task_completion_{timestamp}")
                         save_name=f"MADDPG_task_completion")
    # 能耗均衡曲线
    plotter.plot_rewards([all_fairness],
                         title="UAV Energy Fairness Rate",
                         xlabel="Iteration number",
                         ylabel="Fairness index",
                         smooth=True,
                         window=5,
                         # save_name=f"MADDPG_task_completion_{timestamp}")
                         save_name=f"MADDPG_energy_fairness")

    # 碰撞次数曲线
    '''plotter.plot_rewards(all_collisions,
                         title="Number of Collisions",
                         xlabel="Iteration number",
                         ylabel="Number of Collisions",
                         smooth=True,
                         window=5,
                         save_name=f"MADDPG_collisions_{timestamp}")


    #时间窗强度对比曲线
    plotter.plot_multiple_reward_components(reward_histories,
                            title="UAV Reward Compare Curve",
                            xlabel="Iteration number",
                            ylabel="Reward",
                            smooth=True,
                            window=5,
                            save_name=f"MADDPG_reward_compare_curve_{timestamp}")'''

    # 绘制 UAV 轨迹和任务分布
    plotter.plot_uav_trajectories(maddpg.uav_trajectories, env.task_pos[:, :2],
                                  title="UAV Trajectory and Task Distribution",
                                  save_name=f"MADDPG_trajectory_plot_{time_window_type}")
    # save_name=f"MADDPG_trajectory_plot__{timestamp}")
    # 绘制奖励函数曲线
    plotter.plot_reward_components(env.reward_history)
    print(f"best task rate: {maddpg.best_task_rate}")
    if maddpg.finish_time:
        for k, t in maddpg.finish_time.items():
            print(f"{k}:{t}")
    return all_completion


def test_with_MADDPG(plotter, time_window_type, epochs=500):
    env = UAVEnv(time_window_type)
    maddpg = MAPPO(env)

    all_reward, all_completion, all_collision = maddpg.test(epochs)

    # 绘制奖励曲线
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 奖励曲线
    plotter.plot_rewards([all_reward],
                         title="UAV Reward Curve",
                         xlabel="Iteration number",
                         ylabel="Reward",
                         smooth=True,
                         window=5,
                         save_name=f"reward_curve_{timestamp}")
    # save_name=f"reward_curve")

    # 任务完成率曲线
    plotter.plot_rewards([all_completion],
                         title="UAV Task Completion Rate",
                         xlabel="Iteration number",
                         ylabel="Task completion rate",
                         smooth=True,
                         window=5,
                         save_name=f"task_completion_{timestamp}")

    # 碰撞次数曲线
    '''plotter.plot_rewards(all_collisions,
                         title="Number of Collisions",
                         xlabel="Iteration number",
                         ylabel="Number of Collisions",
                         smooth=True,
                         window=5,
                         save_name=f"collisions_{timestamp}")


    #时间窗强度对比曲线
    plotter.plot_multiple_reward_components(reward_histories,
                            title="UAV Reward Compare Curve",
                            xlabel="Iteration number",
                            ylabel="Reward",
                            smooth=True,
                            window=5,
                            save_name=f"reward_compare_curve_{timestamp}")'''

    # 绘制 UAV 轨迹和任务分布
    plotter.plot_uav_trajectories(env.uav_trajectories, env.task_pos[:, :2],
                                  title="UAV Trajectory and Task Distribution",
                                  # save_name=f"trajectory_plot_{time_window_type}")
                                  save_name=f"trajectory_plot__{timestamp}")

    # 绘制奖励函数曲线
    # plotter.plot_reward_components(env.reward_history)
    return all_reward


def time_window_ablation_experiment(plotter, epochs=500):
    window_types = ['none', 'normal', 'tight']
    labels = ['No Time Window', 'Default Time Window', 'Tight Time Window']

    completion_dict = {}
    for tw_type, label in zip(window_types, labels):
        completion = run_with_MADDPG(plotter, tw_type, epochs)
        completion_dict[label] = [completion]  # 每种设置的所有run的rewards

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plotter.plot_multiple_reward_components(
        completion_dict,
        title="Comparison of Completion Rate Curves under Different Time Window Constraints",
        xlabel="Iteration number",
        ylabel="Completion Rate",
        smooth=True,
        window=5,
        save_name=f"completion_curve_compare_{timestamp}")
    # save_name=f"reward_curve_compare")


def save_completion_data(completion_data, filename="completion_data.json"):
    """保存任务完成率数据到JSON文件"""
    data = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'completion_rates': completion_data
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"数据已保存到 {os.path.abspath(filename)}")


def load_completion_data(filename="completion_data.json"):
    """从JSON文件加载任务完成率数据"""
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"从文件 {filename} 加载数据 (记录时间: {data['timestamp']})")
    return data['completion_rates']


def run_with_GA(plotter, time_window_type, epochs=500):
    solver = GA(
        time_window_type,
        population_size=30,  # 种群数量（可调优）
        generations=epochs,  # 进化代数
        mutation_rate=0.08,  # 变异概率
        elite_fraction=0.2  # 精英比例
    )
    best_plan, completion, timestamp = solver.run()
    # 绘制轨迹图
    solver.visualize(timestamp)
    # 保存数据到文件 (自动添加时间戳)
    save_completion_data(completion, f"GA_completion_data_{timestamp}.json")

    # 从文件读取数据再绘图
    # GA_completion = load_completion_data(f"GA_completion_{timestamp}.json")

    # 任务完成率曲线图
    plotter.plot_rewards(
        rewards=[completion],  # 注意是二维列表形式
        title="GA Task Completion Curve",
        xlabel="Iteration number",
        ylabel="Task completion rate",
        smooth=True,
        window=10,
        # save_name="GA_task_completion")
        save_name=f"GA_task_completion_{timestamp}")


# 训练MAPPO模型并保存模型
def run_with_mappo(plotter, time_window_type, epochs=100):
    env = UAVEnv(time_window_type)
    mappo = MAPPO(env)
    all_reward, all_completion, all_collision, all_energy = mappo.train(epochs)

    # 可视化结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plotter.plot_rewards(
        rewards=[all_completion],  # 注意是二维列表形式
        title="MAPPO Task Completion Curve",
        xlabel="Iteration number",
        ylabel="Task completion rate",
        smooth=True,
        window=10,
        save_name=f"MAPPO_task_completion")
    # save_name = f"MAPPO_task_completion_{timestamp}")
    plotter.plot_uav_trajectories(mappo.uav_trajectories, env.task_pos[:, :2],
                                  # title=f"MAPPO UAV Trajectories", save_name=f"MAPPO_uav_trajectory_{timestamp}")
                                  title=f"MAPPO UAV Trajectories", save_name=f"MAPPO_uav_trajectory")
    print(f"best_task_rate: {mappo.best_task_rate}")
    plotter.plot_rewards(
        rewards=[all_reward],  # 注意是二维列表形式
        title="MAPPO Reward Curve",
        xlabel="Iteration number",
        ylabel="Reward",
        smooth=True,
        window=10,
        save_name=f"MAPPO_reward_curve")
    # save_name = f"MAPPO_reward_curve{timestamp}")


# 测试MAPPO模型
def test_with_mappo(plotter, time_window_type, epochs=100, render=True):
    env = UAVEnv(time_window_type)
    mappo = MAPPO(env)

    all_reward, all_completion, all_collision, all_energy = mappo.test(epochs)

    # 可视化结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plotter.plot_rewards(
        rewards=[all_completion],  # 注意是二维列表形式
        title="MAPPO Task Completion Curve",
        xlabel="Iteration number",
        ylabel="Task completion rate",
        smooth=True,
        window=10,
        save_name=f"MAPPO_task_completion")
    # save_name = f"MAPPO_task_completion_{timestamp}")
    plotter.plot_uav_trajectories(env.uav_trajectories, env.task_pos[:, :2],
                                  # title=f"MAPPO UAV Trajectories", save_name=f"MAPPO_uav_trajectory_{timestamp}")
                                  title=f"MAPPO UAV Trajectories", save_name=f"MAPPO_uav_trajectory")
    plotter.plot_rewards(
        rewards=[all_reward],  # 注意是二维列表形式
        title="MAPPO Reward Curve",
        xlabel="Iteration number",
        ylabel="Reward",
        smooth=True,
        window=10,
        save_name=f"MAPPO_reward_curve")
    # save_name = f"MAPPO_reward_curve{timestamp}")


# def run_with_mappo(plotter, time_window_type, epochs=100):
#     env = UAVEnv(time_window_type)
#     mappo = MAPPO(env)
#
#     all_reward, all_completion, all_collision, all_energy = mappo.train(epochs)
#
#     # === 绘图 ===
#     plotter.plot_rewards(
#         rewards=[all_completion],  # 注意是二维列表形式
#         title="MAPPO Task Completion Curve",
#         xlabel="Iteration number",
#         ylabel="Task completion rate",
#         smooth=True,
#         window=10,
#         # save_name=f"MAPPO_task_completion")
#         save_name = f"GA_task_completion_{timestamp}")
#     plotter.plot_uav_trajectories(env.uav_trajectories, env.task_pos[:, :2],
#                                   title="MAPPO UAV Trajectories", save_name=f"MAPPO_uav_trajectory_{timestamp}")
#     plotter.plot_rewards(
#         rewards = [all_reward],  # 注意是二维列表形式
#         title="MAPPO Reward Curve",
#         xlabel="Iteration number",
#         ylabel="Reward",
#         smooth=True,
#         window=10,
#         # save_name=f"MAPPO_reward_curve")
#         save_name = f"MAPPO_reward_curve{timestamp}")


if __name__ == '__main__':
    # 强制 TensorFlow 使用 GPU（可选）
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)  # 防止显存占满
    print("物理GPU设备:", len(tf.config.list_physical_devices('GPU')))  # 应返回GPU列表
    print("逻辑GPU设备:", tf.config.list_logical_devices('GPU'))  # 应返回GPU列表
    print("GPU是否可用:", tf.test.is_gpu_available())  # 应返回True

    print(sys.executable)  # 输出实际使用的Python路径

    print(os.environ.get('CUDA_VISIBLE_DEVICES'))
    print(os.environ.get('LD_LIBRARY_PATH'))

    print(tf.__version__)  # 打印版本号
    print(tf.config.list_physical_devices('GPU'))  # 检查GPU是否可用
    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 绘图
    plotter = Plotter()
    # MADDPG
    run_with_MADDPG(plotter, 'normal', epochs=300)#300
    # 时间窗消融实验
    # time_window_ablation_experiment(plotter, epochs=300)
    # GA
    # run_with_GA(plotter, 'normal', epochs=100)
    # MAPPO
    # run_with_mappo(plotter, 'normal', epochs=100)
    # run_with_mappo(plotter, 'normal', epochs=2000)
    # test_with_mappo(plotter, 'normal', epochs=10)#不需要过多训练？
