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

    # MADDPG-IL
    # 1) BC（只学任务）
    D_demo = maddpg.collect_expert_data(env, time_window_type, num_episodes=800)  # 800
    maddpg.train_behavior_cloning(D_demo, epochs=20)  # epochs=20

    # 2) DAgger（只标注任务；RL+imit 混合训练；结束后把 imit 关掉）
    maddpg.dagger_training(iterations=20, lambda_im=0.3)  # iterations=20

    # 3) 纯 RL（不再用专家）
    maddpg.lambda_im = 0.0
    maddpg.use_residual = False  # 先关残差；若想进一步优化，再打开
    all_reward, all_completion, all_collision, all_fairness = maddpg.train(epochs)

    #Heuristic
    # all_reward, all_completion, all_collision, all_fairness = maddpg.run_heuristic(epochs)

    # 绘制奖励曲线
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    save_training_data(
        completion_rates=all_completion,
        rewards=all_reward,
        collisions=all_collision,
        fairness=all_fairness,
        # filename=f"MADDPG_data_with_Heuristic_1000_{timestamp}.json"  # TODO
        filename=f"maddpg-il_{timestamp}.json"
    )

    # 奖励曲线
    # if not maddpg.loss:
    #     plotter.plot_loss([maddpg.loss],
    #                       title="UAV Loss曲线",
    #                       xlabel="Iteration number",
    #                       ylabel="Reward",
    #                       smooth=True,
    #                       window=5,
    #                       save_name=f"MADDPG_loss_curve_{timestamp}")
        # save_name=f"MADDPG_loss_curve")
    # 奖励曲线
    plotter.plot_rewards([all_reward],
                         title="UAV奖励曲线",
                         xlabel="Iteration number",
                         ylabel="Reward",
                         smooth=True,
                         window=5,
                         save_name=f"MADDPG_reward_curve_{timestamp}")
    # save_name=f"MADDPG_reward_curve")

    # 任务完成率曲线
    plotter.plot_rewards([all_completion],
                         title="UAV任务完成率",
                         xlabel="Iteration number",
                         ylabel="Task completion rate",
                         smooth=False,
                         window=5,
                         save_name=f"MADDPG_task_completion_{timestamp}")
    # save_name=f"MADDPG_task_completion")
    # 能耗均衡曲线
    plotter.plot_rewards([all_fairness],
                         title="UAV能耗均衡指数",
                         xlabel="Iteration number",
                         ylabel="Fairness index",
                         smooth=False,
                         window=5,
                         save_name=f"MADDPG_energy_fairness_{timestamp}")
    # save_name=f"MADDPG_energy_fairness")

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
                                  env.task_window, maddpg.task_marked,
                                  title="UAV轨迹图",
                                  # save_name=f"MADDPG_trajectory_plot_{time_window_type}")
                                  save_name=f"MADDPG_trajectory_plot_{time_window_type}_{timestamp}")
    # 绘制奖励函数曲线
    plotter.plot_reward_components(env.reward_history)
    print(f"best task rate: {maddpg.best_task_rate}")
    if maddpg.finish_time:
        for k, t in maddpg.finish_time.items():
            print(f"{k}:{t}")
    return all_completion


# def test_with_MADDPG(plotter, time_window_type, epochs=500):
#     env = UAVEnv(time_window_type)
#     maddpg = MADDPG(env)
#
#     all_reward, all_completion, all_collision, all_fairness = maddpg.test(epochs)
#     # 绘制奖励曲线
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     # 奖励曲线
#     plotter.plot_rewards([all_reward],
#                          title="UAV Reward Curve",
#                          xlabel="Iteration number",
#                          ylabel="Reward",
#                          smooth=True,
#                          window=5,
#                          # save_name=f"MADDPG_reward_curve_{timestamp}")
#                          save_name=f"MADDPG_reward_curve")
#
#     # 任务完成率曲线
#     plotter.plot_rewards([all_completion],
#                          title="UAV Task Completion Rate",
#                          xlabel="Iteration number",
#                          ylabel="Task completion rate",
#                          smooth=True,
#                          window=5,
#                          # save_name=f"MADDPG_task_completion_{timestamp}")
#                          save_name=f"MADDPG_task_completion")
#     # 能耗均衡曲线
#     plotter.plot_rewards([all_fairness],
#                          title="UAV Energy Fairness Rate",
#                          xlabel="Iteration number",
#                          ylabel="Fairness index",
#                          smooth=True,
#                          window=5,
#                          # save_name=f"MADDPG_task_completion_{timestamp}")
#                          save_name=f"MADDPG_energy_fairness")
#
#     # 碰撞次数曲线
#     '''plotter.plot_rewards(all_collisions,
#                          title="Number of Collisions",
#                          xlabel="Iteration number",
#                          ylabel="Number of Collisions",
#                          smooth=True,
#                          window=5,
#                          save_name=f"collisions_{timestamp}")
#
#
#     #时间窗强度对比曲线
#     plotter.plot_multiple_reward_components(reward_histories,
#                             title="UAV Reward Compare Curve",
#                             xlabel="Iteration number",
#                             ylabel="Reward",
#                             smooth=True,
#                             window=5,
#                             save_name=f"reward_compare_curve_{timestamp}")'''
#
#     # 绘制 UAV 轨迹和任务分布
#     plotter.plot_uav_trajectories(maddpg.uav_trajectories, env.task_pos[:, :2],
#                                   title="UAV Trajectory and Task Distribution",
#                                   save_name=f"MADDPG_trajectory_plot_{time_window_type}")
#     # save_name=f"MADDPG_trajectory_plot__{timestamp}")
#     # 绘制奖励函数曲线
#     plotter.plot_reward_components(env.reward_history)
#     print(f"best task rate: {maddpg.best_task_rate}")
#     if maddpg.finish_time:
#         for k, t in maddpg.finish_time.items():
#             print(f"{k}:{t}")
#     return all_reward


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


# def save_completion_data(completion_data, filename="completion_data.json"):
#     """保存任务完成率数据到JSON文件"""
#     data = {
#         'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
#         'completion_rates': completion_data
#     }
#     with open(filename, 'w') as f:
#         json.dump(data, f, indent=4)
#     print(f"数据已保存到 {os.path.abspath(filename)}")
#
#
# def load_completion_data(filename="completion_data.json"):
#     """从JSON文件加载任务完成率数据"""
#     with open(filename, 'r') as f:
#         data = json.load(f)
#     print(f"从文件 {filename} 加载数据 (记录时间: {data['timestamp']})")
#     return data['completion_rates']
def save_training_data(completion_rates=None, rewards=None, collisions=None,
                       fairness=None, filename="training_data.json"):
    """
    保存多种训练数据到JSON文件
    参数:
        completion_rates: 任务完成率列表
        rewards: 每轮奖励列表
        collisions: 碰撞次数列表
        fairness: 能耗均衡指数列表
        filename: 保存文件名
    """
    data = {
        'metadata': {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'num_episodes': len(completion_rates) if completion_rates else 0
        },
        'metrics': {
            'completion_rates': completion_rates or [],
            'rewards': rewards or [],
            'collisions': collisions or [],
            'fairness': fairness or []
        }
    }

    # # 如果文件已存在，则读取旧数据并追加新记录
    # if os.path.exists(filename):
    #     with open(filename, 'r') as f:
    #         existing_data = json.load(f)
    #
    #     # 合并数据(假设旧数据也是相同格式)
    #     if isinstance(existing_data, list):
    #         # 旧版本是列表格式，转换为新格式
    #         data = [existing_data, data]
    #     else:
    #         # 新版本是字典格式，创建列表保存历史记录
    #         data = [existing_data, data]

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"训练数据已保存到 {os.path.abspath(filename)}")


def load_training_data(filename="training_data.json"):
    """
    从JSON文件加载训练数据
    返回:
        如果是单次运行数据: 直接返回metrics字典
        如果是多次运行数据: 返回包含所有历史记录的列表
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件 {filename} 不存在")

    with open(filename, 'r') as f:
        data = json.load(f)

    # 处理不同格式的数据
    # if isinstance(data, list):
    #     print(f"从文件 {filename} 加载了 {len(data)} 次训练记录")
    #     return data
    # else:
    #     print(f"从文件 {filename} 加载单次训练数据 (记录时间: {data['metadata']['timestamp']})")
    return data['metrics']


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
    run_with_MADDPG(plotter, 'normal', epochs=1000)  # 1000
    # 模仿学习和纯MADDPG对比实验
    # loaded_data1 = load_training_data("Heuristic_20260113_0122.json")  # Heuristic
    # loaded_data3 = load_training_data("MADDPG_pure_20260112_0153.json")  # MADDPG_data_with_pure
    # loaded_data4 = load_training_data("MADDPG_data_with_Heuristic_1000_20260106_0014.json")

    #无人机数量对比试验
    # loaded_data1 = load_training_data("MADDPG_IL_3uav_20260117_0025.json")  # Heuristic
    # loaded_data6 = load_training_data("Heuristic_2uav_20260119_2253.json")  # Heuristic
    # loaded_data3 = load_training_data("MADDPG_IL_3uav_20260117_0229.json")  # MADDPG_data_with_pure
    # loaded_data5 = load_training_data("Heuristic_2uav_20260119_2326.json")  # MADDPG_data_with_pure
    # loaded_data4 = load_training_data("MADDPG_data_with_Heuristic_1000_20260106_0014.json")
    # loaded_data7 = load_training_data("Heuristic_2uav_20260120_0044.json")  # MADDPG_data_with_pure
    #
    loaded_data1 = load_training_data("maddpg - il_20260120_2041.json")  # 硬时间窗

    #
    # completion_dict = {"MADDPG-IL 4 uavs": [loaded_data4['completion_rates']],
    #                    "MADDPG-IL 3 uavs": [loaded_data1['completion_rates']],
    #                    "MADDPG-IL 2 uavs": [loaded_data3['completion_rates']],
    #                    # "Heuristic 4 uavs": [loaded_data7['completion_rates']],
    #                    # "Heuristic 3 uavs": [loaded_data6['completion_rates']],
    #                    #  "Heuristic 2 uavs": [loaded_data5['completion_rates']],
    #                    }
    #
    # plotter.plot_multiple_reward_components(
    #     completion_dict,
    #     title="Comparison of Completion Rate Curves under Different Algorithm",
    #     xlabel="Iteration number",
    #     ylabel="Completion Rate",
    #     smooth=True,
    #     window=5,
    #     save_name=f"completion_curve_compare_{timestamp}")
    # #
    # 时间窗消融实验
    # time_window_ablation_experiment(plotter, epochs=300)
