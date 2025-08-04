import numpy as np
from config import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
from scipy import stats
import os
import math

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class UAVEnv:
    def __init__(self, time_window_type):
        self.num_uav = UAV_NUM
        self.num_task = TASK_NUM
        self.T = TIME_STEPS
        self.t = 0

        # 验证时间窗约束
        # task_pos_2d = np.array([
        #     [38.24, 33.82], [30.98, 56.18], [50, 70], [69.02, 56.18], [61.76, 33.82]
        # ], dtype=np.float32)
        task_pos_2d = np.array([
            [20, 28], [25, 23], [38, 30], [75, 25], [80, 40],
            [20, 40], [35, 55], [50, 55], [70, 50], [78, 60],
            [30, 71], [45, 78], [25, 80], [75, 75], [80, 69]
        ], dtype=np.float32)
        self.task_pos = np.hstack((task_pos_2d, np.zeros((self.num_task, 1))))

        # 验证时间窗约束
        # init_positions = [
        #                      [20, 56]
        #                  ][:self.num_uav]
        init_positions = [
                             [20, 90], [85, 15], [15, 15]
                         ][:self.num_uav]
        self.uav_pos = np.array(init_positions, dtype=np.float32)

        # self.task_window = [(5, 90) for _ in range(self.num_task)]
        self.all_tasks_done = False
        self.task_done = np.zeros(self.num_task)
        self.task_marked = np.zeros(self.num_task, dtype=bool)  # 标记任务是否完成Bool
        # self.energy = np.zeros(self.num_uav)
        self.energy = np.full(self.num_uav, E_MAX * 0.01)  # 初始化为1%能量而非0
        self.load = np.zeros(self.num_uav)
        self.count_collisions = 0  # 碰撞次数
        self.tasks_completed = 0.0  # 任务完成率
        self.all_tasks_done_time = -1  # 全部任务完成时间
        self.reward_history = {'R_S': [], 'R_D': [], 'R_E': [], 'R_C': [], 'R_global': [],
                               'R_attract': [], 'R_border': [], 'R_TW': []}  # 记录奖励函数曲线

        self.H = H
        self.v_max = V_MAX
        self.E_max = E_MAX
        self.d_safe = D_SAFE
        self.d_max = D_MAX
        self.xi = XI
        self.lambda_ = LAMBDA
        self.G = G
        self.W = W
        self.p_th = P_TH

        self.a1 = A1
        self.a2 = A2
        self.a3 = A3
        self.a4 = A4
        self.a5 = A5
        self.a6 = A6
        self.a7 = A7
        self.a8 = A8
        self.a9 = A9

        self.alpha = 3.0
        self.beta = 2.0

        self.uav_trajectories = [[] for _ in range(self.num_uav)]
        for i in range(self.num_uav):
            self.uav_trajectories[i].append(init_positions[i].copy())  # 记录轨迹
        self.dt = D_T

        self.time_window_type = time_window_type
        # 时间窗对比实验
        if time_window_type == 'none':
            self.task_window = [(0, self.T) for _ in range(self.num_task)]  # 全时间段有效
        elif time_window_type == 'tight':
            # self.task_window = [(self.T // 2 - 5, self.T // 2 + 5) for _ in range(self.num_task)]  # 紧时间窗，跨度10
            # self.task_window = [(i, i + 30 - 1) for i in range(50, TIME_STEPS, 30)] #50~500区间，每个任务窗口期为30,紧锣密鼓的挨着
            self.task_window = [(0, 30), (30, 50), (50, 65), (65, 80), (80, 100)]
        else:
            # self.task_window = [(5, 90) for _ in range(self.num_task)]  # 默认情况
            self.task_window = []
            min_window, max_window = 30, 60  # 窗口大小范围
            for _ in range(self.num_task):
                # 随机窗口大小（30-60）
                win_size = np.random.randint(min_window, max_window + 1)
                # 随机起始时间（确保窗口不超出总时长）
                start = np.random.randint(0, self.T - win_size + 1)
                self.task_window.append((start, start + win_size))

        self.finish_time = {k: 0 for k in range(self.num_task)}  # 记录任务完成时间

    def reset(self, time_window_type):
        self.__init__(time_window_type)
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.num_uav):
            uav_obs = list(self.uav_pos[i]) + [self.energy[i], self.t / self.T]

            # 加入 UAV 编号归一化信息（例如 UAV 0 → 0.0，UAV 4 → 0.8）
            uav_obs += [i / (self.num_uav - 1 + 1e-8)]

            for task in self.task_pos:
                uav_obs += list(task[:2])
            for (start, end) in self.task_window:
                uav_obs += [(self.t - (start + end) / 2) / self.T]
            uav_obs += list(self.task_done / self.p_th)
            obs.append(uav_obs)
        return obs

    def step(self, actions):
        rewards = []
        prev_pos = self.uav_pos.copy()  # 记录每架 UAV 当前的位置，用于后续计算移动距离
        p_individual = np.zeros((self.num_uav, self.num_task))  # 每个 UAV 对每个任务的感知概率
        delta = np.zeros((self.num_uav, self.num_task))  # 每个 UAV 对每个任务的时间窗调制值
        task_participants = {j: [] for j in range(self.num_task)}  # 每个任务的参与 UAV 列表

        for i, (v, theta, k) in enumerate(actions):
            k = int(k)
            # 获取未完成的任务编号
            unfinished = [j for j in range(self.num_task) if not self.task_marked[j]]
            withinTW = [j for j in unfinished if self.t < self.task_window[j][1]]
            # 处理当前所有未完成任务都已超时的情况
            if not withinTW:
                return self._get_obs(), np.zeros(self.num_uav), True

            # # 如果选择的任务已完成，则改为最近的未完成任务
            # if unfinished and self.task_marked[int(k)]:
            #     dists = [distance(self.uav_pos[i], self.task_pos[j][:2]) for j in unfinished]
            #     k = unfinished[np.argmin(dists)]  # 选择最近的未完成任务

            # 2. 强制重新定向到最紧急任务（如果当前目标非最紧急）
            # 计算全局最紧急任务（剩余时间最少）
            # most_urgent = min(withinTW, key=lambda j: remain_time(self.t, self.task_window[j]))
            # if k != most_urgent and distance(self.uav_pos[i], self.task_pos[k][:2]) > self.d_max:
            #     k = most_urgent  # 切换到最紧急任务
            #     theta = np.arctan2(self.task_pos[k][1] - self.uav_pos[i][1],
            #                        self.task_pos[k][0] - self.uav_pos[i][0])  # 直接朝向新目标

            # V3.0
            scores = []
            reward = 0.0
            for j in withinTW:
                urgency = 1.0 - remain_time(self.t, self.task_window[j])
                closest = distance(self.uav_pos[i], self.task_pos[j][:2])
                # scores.append(urgency / (closest + 1e-5))
                scores.append(urgency / (closest + 1) + (self.num_task - len(unfinished)) * 0.2)
            optimal_task = withinTW[np.argmax(scores)]
            # if if_self_learn:
            #     k = optimal_task
            # elif k == optimal_task:
            #     reward += 20.0#10.0
            # 计算与任务点的距离
            dist = distance(self.uav_pos[i], self.task_pos[k][:2])
            if k != optimal_task and dist > self.d_max:
                k = optimal_task  # 切换到最紧急任务
                theta = np.arctan2(self.task_pos[k][1] - self.uav_pos[i][1],
                                   self.task_pos[k][0] - self.uav_pos[i][0])

            # 基于动作更新 UAV 的位置（带限位）
            dx = v * np.cos(theta) * self.dt
            dy = v * np.sin(theta) * self.dt
            self.uav_pos[i][0] = np.clip(self.uav_pos[i][0] + dx, 0, MAP_SIZE)
            self.uav_pos[i][1] = np.clip(self.uav_pos[i][1] + dy, 0, MAP_SIZE)

            # 记录 UAV 的新位置以用于绘图
            self.uav_trajectories[i].append(self.uav_pos[i].copy())  # 记录轨迹


            # 计算 UAV 的感知概率（超出最大感知距离则为 0）
            # P = np.exp(-self.xi * dist) if dist <= self.d_max else 0.0
            P = 1 if dist <= self.d_max else 0.0  # TODO

            # 时间窗调制项，越靠近中心时间越高
            start_t, end_t = self.task_window[k]
            center_t = (start_t + end_t) / 2
            # delta_k = np.exp(-self.lambda_ * abs(self.t - center_t)) if start_t <= self.t <= end_t else 0.0
            delta_k = 1 if start_t <= self.t <= end_t else 0.0  # TODO

            # 保存感知精度与时间窗调制系数
            p_individual[i, k] = P
            delta[i, k] = delta_k

            # 如果在感知范围内，加入该任务的参与 UAV 列表
            if dist <= self.d_max and start_t <= self.t <= end_t:
                task_participants[k].append(i)

            # 计算 UAV 的能耗（飞行+感知）
            move_dist = distance(prev_pos[i], self.uav_pos[i])
            # E_f = 0.5 * self.G * (move_dist / self.dt) ** 2 * self.dt  # 飞行能耗
            E_f = 0.3 * self.G * move_dist  # 改为线性关系（原平方关系放大差异）
            E_s = self.W if dist <= self.d_max else 0.0  # 感知能耗
            # self.energy[i] += E_f + E_s  # 累加总能耗
            self.energy[i] = np.clip(self.energy[i] + E_f + E_s, 0.01 * E_MAX, E_MAX)

            # 计算奖励各项分量
            R_S = delta_k * P  # 时间窗调制的感知成功概率奖励
            R_D = np.exp(-self.xi * dist)
            # R_D = 0.5 * np.exp(-self.xi * dist) + 0.5 / (1.0 + self.xi * dist)  # 目标靠近 shaping 奖励

            R_C = self._calculate_collision_reward_single(i)  # 碰撞惩罚
            R_E = E_f + E_s  # 能耗惩罚

            # 吸引任务点：鼓励靠近未完成任务
            # R_attract = -min([distance(self.uav_pos[i], self.task_pos[k][:2])
            #                   for k in unfinished], default=0) / MAP_SIZE
            # temp = min([distance(self.uav_pos[i], self.task_pos[k][:2])
            #                   for k in unfinished], default=0)
            # R_attract = 1.0 / (1.0 + temp) * 5.0

            # weighted_dists = []
            # for j in withinTW:
            #     a = 1.0 - remain_time(self.t, self.task_window[j] )
            #     b = distance(self.uav_pos[i], self.task_pos[j][:2])
            #     weighted_dists.append(a * b)  # 紧急度越高、距离影响越大
            # min_weighted_dist = min(weighted_dists)
            # R_attract = 10.0 / (1.0 + min_weighted_dist)  # 放大奖励梯度，让靠近紧急任务更“赚”

            # 计算当前选择任务(k)的吸引力
            chosen_task_urgency = 1.0 - remain_time(self.t, self.task_window[k])
            chosen_task_dist = distance(self.uav_pos[i], self.task_pos[k][:2])
            R_attract = 10.0 * (chosen_task_urgency / (chosen_task_dist + 1.0))

            # # 运动方向奖励
            prev_dist = distance(prev_pos[i], self.task_pos[k][:2])
            # # 动态方向奖励（基于距离变化比例）
            # dist_ratio = (prev_dist - dist) / prev_dist  # 距离缩短的比例
            # R_direction = 10.0 * dist_ratio if dist_ratio > 0 else -1.0 * abs(dist_ratio)
            R_direction = 0
            # dist_improvement = prev_dist - dist
            # R_direction = np.tanh(dist_improvement) * 5.0
            # R_direction = 20.0 * np.tanh(dist_improvement / 5.0)

            # 剩余时间权重
            R_TW = 1.0 - remain_time(self.t, self.task_window[k][:2])

            # 边界惩罚
            R_border = -5.0 if (self.uav_pos[i][0] <= 0 or self.uav_pos[i][0] >= MAP_SIZE or
                                self.uav_pos[i][1] <= 0 or self.uav_pos[i][1] >= MAP_SIZE) else 0.0

            # 在每步奖励中添加全局进度分量
            global_progress = np.sum(
                self.task_marked) / self.num_task  # 但是这样会导致无人机更愿意停留在已完成任务的区域，获得高R_Global奖励，而非探索新的区域
            R_Global = 80.0 * global_progress  # 40# 整体进度奖励

            # 综合计算当前 UAV 的总奖励
            reward += (self.a1 * R_S + self.a2 * R_D - self.a3 * R_E / E_MAX - self.a4 * R_C + self.a6 * R_attract
                       + self.a7 * R_border + self.a9 * R_direction + 0 * R_Global)
            # 轻微惩罚无效选择
            if dist > self.d_max:
                # 惩罚与距离成正比，同时考虑 UAV 是否在远离任务点
                penalty = -0.5 * (dist / self.d_max) if dist >= prev_dist else -0.1 * (dist / self.d_max)
                reward += penalty
            rewards.append(reward)

            # 群体感知判定与任务完成,在群体任务完成判定后
        p_group = 0
        for j in withinTW:
            if len(task_participants[j]) > 0:
                ps = [1 - p_individual[i, j] for i in task_participants[j]]
                p_group = 1 - np.prod(ps)
                p_group *= delta_k  # 时间窗进行调制
                if p_group >= self.p_th:
                    self.task_done[j] = p_group
                    self.task_marked[j] = True
                    self.tasks_completed += 1  # 记录已经完成的任务数量
                    self.finish_time[j] = self.t
                    # 平均奖励发放给参与者
                    for i in task_participants[j]:
                        rewards[i] += self.a5 * p_group / self.num_uav  # 降低主力 UAV 奖励，但可提升整体协作性

        # Jain公平指数计算
        fairness = self.calculate_fairness()  # 能耗均衡
        # 公平性奖励（范围[-1,1]）
        # R_Fairness = 2.0 * (fairness - 0.5)  # fairness越接近1越平均
        # 非线性奖励映射（增强均衡激励）
        R_Fairness = 5.0 * (np.tanh(4.0 * (fairness - 0.7)) + 1)  # 范围[0,10]，0.7为阈值

        for j in range(len(rewards)):
            rewards[j] += self.a8 * R_Fairness

        # 绘制奖励函数曲线
        self.reward_history['R_S'].append(self.a1 * R_S)
        self.reward_history['R_D'].append(self.a2 * R_D)
        # self.reward_history['R_E'].append(R_E)
        self.reward_history['R_C'].append(self.a4 * R_C)
        self.reward_history['R_global'].append(self.a5 * p_group / self.num_uav)
        # self.reward_history['R_attract'].append(self.a6 * R_attract)
        self.reward_history['R_border'].append(self.a7 * R_border)
        self.reward_history['R_TW'].append(self.a8 * R_TW)

        self._count_collisions(i)  # 检查这次是否有碰撞

        # 检查是否所有任务都已完成
        self.all_tasks_done = np.all(self.task_marked)
        if self.all_tasks_done:
            self.all_tasks_done_time = self.t
        done = self.t > self.T or self.all_tasks_done

        # 所有任务都被完成
        if done:
            return self._get_obs(), np.array(rewards, dtype=np.float32), done
        else:
            self.t += 1
        return self._get_obs(), np.array(rewards, dtype=np.float32), done

    # 碰撞奖励
    def _calculate_collision_reward_single(self, i):
        reward = 0.0
        for j in range(self.num_uav):
            if i != j:
                d_ij = distance(self.uav_pos[i], self.uav_pos[j])
                if d_ij < self.d_safe:
                    # reward += 1.0
                    reward += 5.0  # 1.0
        return reward

    # 碰撞次数
    def _count_collisions(self, i):
        for j in range(i + 1, self.num_uav):
            d = distance(self.uav_pos[i], self.uav_pos[j])
            if d < self.d_safe:
                self.count_collisions += 1

    # 能耗均衡
    def calculate_fairness(self):
        # energy_norm = self.energy / (E_MAX + 1e-8)
        # return (np.sum(energy_norm) ** 2) / (self.num_uav * np.sum(energy_norm ** 2))

        temp = 0
        for i in range(len(self.energy)):
            temp += self.energy[i] ** 2
        fairness = (np.sum(self.energy) ** 2) / (self.num_uav * temp + 1e-16)
        return fairness
        # 安全计算energy_norm
        # energy_clipped = np.clip(self.energy, 0, E_MAX)
        # energy_norm = energy_clipped / (E_MAX + 1e-8)
        #
        # # 处理全零特殊情况
        # if np.all(energy_norm < 1e-8):
        #     return 0.0  # 或返回1.0取决于设计需求
        #
        # # 稳定计算Jain指数
        # sum_en = np.sum(energy_norm)
        # sum_sq = np.sum(energy_norm ** 2)
        # fairness = (sum_en ** 2) / (self.num_uav * sum_sq + 1e-16)
        # return fairness
