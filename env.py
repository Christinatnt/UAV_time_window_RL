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
        #     [82.4, 156.8], [234.1, 876.3], [456.7, 234.5], [678.9, 567.1], [123.4, 789.0],
        #     [345.6, 123.4], [567.8, 345.6], [789.0, 678.9], [912.3, 234.5], [156.7, 456.7],
        #     [432.1, 789.0], [654.3, 123.4], [876.5, 456.7], [198.7, 654.3], [371.0, 787.6]
        # ], dtype=np.float32)
        task_pos_2d = np.array([
            # 组1：简单任务
            [200, 200],
            [200, 800],
            [800, 200],
            [800, 800],

            # 组2：中期任务，展示时间窗驱动
            [350, 250],
            [250, 650],
            [650, 250],
            [650, 650],

            # 组3：展示协同
            [400, 450],
            [450, 500],
            [600, 550],

            # 组4：后期战略任务
            [300, 500],
            [500, 300],
            [700, 500],
            [500, 700],
        ], dtype=np.float32)
        self.task_pos = np.hstack((task_pos_2d, np.zeros((self.num_task, 1))))

        self.task_window = [
            # 组1：简单任务
            [0, 25],
            [1, 30],
            [0, 35],
            [0, 35],

            # 组2：中期任务，展示时间窗驱动
            [15, 60],
            [12, 55],
            [10, 63],
            [9, 57],

            # 组3：展示协同
            [30, 79],
            [30, 82],
            [30, 75],

            # 组4：后期战略任务
            [45, 100],
            [50, 90],
            [55, 97],
            [56, 95],
        ]

        # 验证时间窗约束
        # init_positions = [
        #                      [50, 950], [950, 50], [50, 50]
        #                  ][:self.num_uav]
        init_positions = [
                             [100, 100],  # UAV1: 左下角
                             [100, 900],  # UAV2: 左上角
                             [900, 100],  # UAV3: 右下角
                             # [900, 900]
                         ][:self.num_uav]
        self.uav_pos = np.array(init_positions, dtype=np.float32)

        self.all_tasks_done = False
        self.task_done = np.zeros(self.num_task)
        self.task_marked = np.zeros(self.num_task, dtype=bool)  # 标记任务是否完成Bool
        # self.energy = np.zeros(self.num_uav)
        self.energy = np.full(self.num_uav, E_MAX * 0.01)  # 初始化为1%能量而非0
        self.load = np.zeros(self.num_uav)
        self.count_collisions = 0  # 碰撞次数
        self.all_tasks_done_time = -1  # 全部任务完成时间
        self.reward_history = {'R_S': [], 'R_D': [], 'R_C': [], 'R_global': [],
                               'R_border': [], 'R_Fairness': []}  # 记录奖励函数曲线

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
        # if time_window_type == 'none':
        #     self.task_window = [(0, self.T) for _ in range(self.num_task)]  # 全时间段有效
        # elif time_window_type == 'tight':
        #     # self.task_window = [(self.T // 2 - 5, self.T // 2 + 5) for _ in range(self.num_task)]  # 紧时间窗，跨度10
        #     # self.task_window = [(i, i + 30 - 1) for i in range(50, TIME_STEPS, 30)] #50~500区间，每个任务窗口期为30,紧锣密鼓的挨着
        #     self.task_window = [(0, 30), (30, 50), (50, 65), (65, 80), (80, 100)]
        # else:
        #     # self.task_window = [(5, 90) for _ in range(self.num_task)]  # 默认情况
        #     self.task_window = []
        #     min_window, max_window = 60, 150  # 窗口大小范围
        #     for _ in range(self.num_task):
        #         # 随机窗口大小（30-60）
        #         win_size = np.random.randint(min_window, max_window + 1)
        #         # 随机起始时间（确保窗口不超出总时长）
        #         start = np.random.randint(0, self.T - win_size + 1)
        #         self.task_window.append((start, start + win_size))

        self.finish_time = {k: 0 for k in range(self.num_task)}  # 记录任务完成时间

    def reset(self, time_window_type):
        self.__init__(time_window_type)
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.num_uav):
            # UAV位置归一化到[0,1]，能量也归一化
            uav_obs = [self.uav_pos[i][0] / MAP_SIZE,
                       self.uav_pos[i][1] / MAP_SIZE,
                       self.energy[i] / E_MAX,
                       self.t / self.T]

            # UAV编号归一化
            uav_obs += [i / (self.num_uav - 1 + 1e-8)]

            # 任务位置归一化
            for task in self.task_pos:
                uav_obs += [task[0] / MAP_SIZE, task[1] / MAP_SIZE]

            # 时间窗归一化（中心偏移/T）
            for (start, end) in self.task_window:
                uav_obs += [(self.t - (start + end) / 2) / self.T]

            # 任务完成度（概率阈值归一化）
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
            withinTW = [j for j in unfinished if self.t <= self.task_window[j][1]]
            # 处理当前所有未完成任务都已超时的情况
            if not withinTW:
                return self._get_obs(), np.zeros(self.num_uav), True
            reward = 0.0

            # scores = []
            # if withinTW:
            #     for j in withinTW:
            #         urgency = 1.0 - remain_time(self.t, self.task_window[j])
            #         closest = distance(self.uav_pos[i], self.task_pos[j][:2])
            #         scores.append(urgency / (closest + 1e-5))
            #         # scores.append(urgency / (closest + 1) + (self.num_task - len(unfinished)) * 0.2)
            #     optimal_task = withinTW[np.argmax(scores)]
            #     # 计算与任务点的距离
            #     dist = distance(self.uav_pos[i], self.task_pos[k][:2])
            #     if k != optimal_task and dist > self.d_max:
            #         k = optimal_task  # 切换到最紧急任务
            #         theta = np.arctan2(self.task_pos[k][1] - self.uav_pos[i][1],
            #                            self.task_pos[k][0] - self.uav_pos[i][0])
            #
            # candidates = []
            # for j in unfinished:
            #     if self.t <= self.task_window[j][1]:  # 未完成且未超时
            #         t_start, t_end = self.task_window[j]
            #         dist = distance(self.uav_pos[i], self.task_pos[j][:2])
            #
            #         # 动态权重：时间窗越临近开启，紧迫性权重越高
            #         if self.t < t_start:
            #             time_ratio = (t_start - self.t) / (t_end - t_start)  # 距离时间窗开启的倒计时
            #             urgency = np.exp(-time_ratio)  # 越接近t_start，urgency越高
            #             score = 0.3 * urgency + 0.7 * (1 / (dist + 1e-5))  # 侧重距离
            #         else:
            #             remain = (t_end - self.t) / (t_end - t_start+1)  # 剩余时间比例
            #             urgency = 1.0 - remain
            #             score = 0.7 * urgency + 0.3 * (1 / (dist + 1e-5))  # 侧重紧迫性
            #
            #         candidates.append((score, j))
            #
            # if candidates:
            #     optimal_task = max(candidates)[1]# 切换到最紧急任务
                # theta = np.arctan2(self.task_pos[k][1] - self.uav_pos[i][1],
                #                    self.task_pos[k][0] - self.uav_pos[i][0])

            #     if k == optimal_task:
            #         reward += 100.0

            # 在 reward 中加入 soft scoring 引导
            urgency = 1.0 - remain_time(self.t, self.task_window[k])
            closest = distance(self.uav_pos[i], self.task_pos[k][:2])
            shaping = urgency * closest

            dist = distance(self.uav_pos[i], self.task_pos[k][:2])

            # 基于动作更新 UAV 的位置（带限位）
            dx = v * np.cos(theta) * self.dt
            dy = v * np.sin(theta) * self.dt
            self.uav_pos[i][0] = np.clip(self.uav_pos[i][0] + dx, 0, MAP_SIZE)
            self.uav_pos[i][1] = np.clip(self.uav_pos[i][1] + dy, 0, MAP_SIZE)

            # 记录 UAV 的新位置以用于绘图
            self.uav_trajectories[i].append(self.uav_pos[i].copy())  # 记录轨迹

            # 计算 UAV 的感知概率（超出最大感知距离则为 0）
            P = np.exp(-self.xi * dist) if dist <= self.d_max else 0.0
            # P = 1 if dist <= self.d_max else 0.0  # TODO

            # 时间窗调制项，越靠近中心时间越高
            start_t, end_t = self.task_window[k]
            center_t = (start_t + end_t) / 2
            # delta_k = np.exp(-self.lambda_ * abs(self.t - center_t))  # 软时间窗
            delta_k = 1 if start_t <= self.t <= end_t else 0.0  #硬时间窗

            # 使用平滑的分段函数
            dist_to_center = abs(self.t - center_t)
            boundary_penalty = max(start_t - self.t, 0, self.t - end_t)
            # delta_k = 1.0#np.exp(-self.lambda_ * (dist_to_center + 10 * boundary_penalty))

            # 保存感知精度与时间窗调制系数
            p_individual[i, k] = P
            delta[i, k] = delta_k

            # 如果在感知范围内，加入该任务的参与 UAV 列表
            if dist <= self.d_max :#and start_t <= self.t <= end_t:
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
            R_D = np.exp(-self.xi * dist) if v != 0 else 0
            # R_D = 0.5 * np.exp(-self.xi * dist) + 0.5 / (1.0 + self.xi * dist)  # 目标靠近 shaping 奖励

            R_C = self._calculate_collision_reward_single(i)  # 碰撞惩罚
            R_E = E_f + E_s  # 能耗惩罚

            # # 运动方向奖励
            # dtheta = abs(theta[i]-prev_theta[i])
            # R_smooth = -1 * dtheta

            # 边界惩罚
            R_border = -5.0 if (self.uav_pos[i][0] <= 0 or self.uav_pos[i][0] >= MAP_SIZE or
                                self.uav_pos[i][1] <= 0 or self.uav_pos[i][1] >= MAP_SIZE) else 0.0

            # 在每步奖励中添加全局进度分量，但是这样会导致无人机更愿意停留在已完成任务的区域，获得高R_Global奖励，而非探索新的区域

            # 综合计算当前 UAV 的总奖励
            reward += (self.a1 * R_S + self.a2 * R_D - self.a3 * R_E / E_MAX - self.a4 * R_C +
                       self.a7 * R_border + 0.1 * shaping)  # + self.a6*R_smooth)
            rewards.append(reward)

            # 群体感知判定与任务完成,在群体任务完成判定后
        p_group = 0
        for j in withinTW:
            if len(task_participants[j]) > 0:
                for i in task_participants[j]:
                    ps = [1 - p_individual[i, j]]
                    p_group = (1 - np.prod(ps)) * delta[i, j]  # 时间窗进行调制
                    if p_group >= self.p_th:
                        self.task_done[j] = p_group
                        self.task_marked[j] = True
                        self.finish_time[j] = self.t
                        # 平均奖励发放给参与者
                        for i in task_participants[j]:
                            rewards[i] += self.a5 * p_group / self.num_uav  # 降低主力 UAV 奖励，但可提升整体协作性

        # Jain公平指数计算
        fairness = self.calculate_fairness()  # 能耗均衡
        # 公平性奖励（范围[-1,1]）
        # R_Fairness = 2.0 * (fairness - 0.5)  # fairness越接近1越平均
        # 非线性奖励映射（增强均衡激励）
        R_Fairness = 1.0 * (np.tanh(4.0 * (fairness - 0.7)) + 1)  # 范围[0,10]，0.7为阈值

        for j in range(len(rewards)):
            rewards[j] += self.a8 * R_Fairness

        # 绘制奖励函数曲线
        self.reward_history['R_S'].append(self.a1 * R_S)
        self.reward_history['R_D'].append(self.a2 * R_D)
        self.reward_history['R_C'].append(self.a4 * R_C)
        self.reward_history['R_global'].append(self.a5 * p_group / self.num_uav)
        self.reward_history['R_border'].append(self.a7 * R_border)
        self.reward_history['R_Fairness'].append(self.a8 * R_Fairness)

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
        temp = 0
        for i in range(len(self.energy)):
            temp += self.energy[i] ** 2
        fairness = (np.sum(self.energy) ** 2) / (self.num_uav * temp + 1e-16)
        return fairness
