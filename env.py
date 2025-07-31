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

        # 验证紧时间窗
        # task_pos_2d = np.array([
        #     [40, 35], [35, 55], [50, 65], [65, 55], [60, 35]
        # ], dtype=np.float32)
        task_pos_2d = np.array([
            [38.24, 33.82], [30.98, 56.18], [50, 70], [69.02, 56.18], [61.76, 33.82]
        ], dtype=np.float32)
        # task_pos_2d = np.array([
        #     [20, 28], [25, 23], [38, 30], [75, 25], [80, 40],
        #     [20, 40], [35, 55], [50, 55], [70, 50], [78, 60],
        #     [30, 71], [45, 78], [25, 80], [75, 75], [80, 69]
        # ], dtype=np.float32)
        self.task_pos = np.hstack((task_pos_2d, np.zeros((self.num_task, 1))))

        # 验证紧时间窗
        init_positions = [
                             [50, 50]
                         ][:self.num_uav]
        # init_positions = [
        #                      [20, 90], [85, 15], [15, 15]
        #                  ][:self.num_uav]
        self.uav_pos = np.array(init_positions, dtype=np.float32)

        # self.task_window = [(5, 90) for _ in range(self.num_task)]
        self.all_tasks_done = False
        self.task_done = np.zeros(self.num_task)
        self.task_marked = np.zeros(self.num_task, dtype=bool)  # 标记任务是否完成Bool
        self.energy = np.zeros(self.num_uav)
        self.load = np.zeros(self.num_uav)
        self.last_task_choice = [-1] * self.num_uav
        self.reward_history = {'R_S': [], 'R_D': [], 'R_E': [], 'R_C': [], 'R_global': [],
                               'R_attract': [], 'R_border': [], 'R_TW': []}  # 记录奖励函数曲线

        self.H = H
        self.v_max = V_MAX
        # self.P_req = P_REQ
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
            self.task_window = [(5, 90) for _ in range(self.num_task)]  # 默认情况

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
            # 获取未完成的任务编号
            unfinished = [j for j in range(self.num_task) if not self.task_marked[j]]
            withinTW = [j for j in unfinished if self.t < self.task_window[j][1]]
            # 处理当前所有未完成任务都已超时的情况
            if not withinTW:
                return self._get_obs(), np.zeros(self.num_uav), False, {
                    'tasks_completed': np.sum(self.task_marked) / self.num_task,
                    'energy': self.energy.copy(),
                    'collision': self._count_collisions(),
                }

            # 计算全局最紧急任务（剩余时间最少）
            most_urgent = min(withinTW, key=lambda j: remain_time(self.t, self.task_window[j]))

            # # 如果选择的任务已完成，则改为最近的未完成任务
            # if unfinished and self.task_marked[int(k)]:
            #     dists = [distance(self.uav_pos[i], self.task_pos[j][:2]) for j in unfinished]
            #     k = unfinished[np.argmin(dists)]  # 选择最近的未完成任务

            # 2. 强制重新定向到最紧急任务（如果当前目标非最紧急）
            if k != most_urgent and distance(self.uav_pos[i], self.task_pos[k][:2]) > self.d_max:
                k = most_urgent  # 切换到最紧急任务
                theta = np.arctan2(self.task_pos[k][1] - self.uav_pos[i][1],
                                   self.task_pos[k][0] - self.uav_pos[i][0])  # 直接朝向新目标

            # 基于动作更新 UAV 的位置（带限位）
            dx = v * np.cos(theta) * self.dt
            dy = v * np.sin(theta) * self.dt
            self.uav_pos[i][0] = np.clip(self.uav_pos[i][0] + dx, 0, MAP_SIZE)
            self.uav_pos[i][1] = np.clip(self.uav_pos[i][1] + dy, 0, MAP_SIZE)

            # 记录 UAV 的新位置以用于绘图
            self.uav_trajectories[i].append(self.uav_pos[i].copy())  # 记录轨迹

            # 计算与任务点的距离
            dist = distance(self.uav_pos[i], self.task_pos[k][:2])
            # 计算 UAV 的感知概率（超出最大感知距离则为 0）
            P = np.exp(-self.xi * dist) if dist <= self.d_max else 0.0

            # 时间窗调制项，越靠近中心时间越高
            start_t, end_t = self.task_window[k]
            center_t = (start_t + end_t) / 2
            delta_k = np.exp(-self.lambda_ * abs(self.t - center_t)) if start_t <= self.t <= end_t else 0.0

            # 保存感知精度与时间窗调制系数
            p_individual[i, k] = P
            delta[i, k] = delta_k

            # 如果在感知范围内，加入该任务的参与 UAV 列表
            if dist <= self.d_max and start_t <= self.t <= end_t:
                task_participants[k].append(i)

            # 计算 UAV 的能耗（飞行+感知）
            move_dist = distance(prev_pos[i], self.uav_pos[i])
            E_f = 0.5 * self.G * (move_dist / self.dt) ** 2 * self.dt  # 飞行能耗
            E_s = self.W if dist <= self.d_max else 0.0  # 感知能耗
            self.energy[i] += E_f + E_s  # 累加总能耗

            # 计算奖励各项分量
            R_S = delta_k * P  # 时间窗调制的感知成功概率奖励
            # R_D = np.exp(-self.xi * dist)
            R_D = 0.5 * np.exp(-self.xi * dist) + 0.5 / (1.0 + self.xi * dist)  # 目标靠近 shaping 奖励
            R_C = self._calculate_collision_reward_single(i)  # 碰撞惩罚
            R_E = E_f + E_s  # 能耗惩罚

            # 吸引任务点：鼓励靠近未完成任务
            # R_attract = -min([distance(self.uav_pos[i], self.task_pos[k][:2])
            #                   for k in unfinished], default=0) / MAP_SIZE

            # time_remaining = {}
            # for j in withinTW:
            #     time_remaining[j] = remain_time(self.t, self.task_window[j])
            # # 找到时间最紧迫的任务（剩余时间比例最小的）
            # most_urgent_idx = np.argmin(time_remaining)  # argmin返回数组中最小元素的索引
            # most_urgent_task = withinTW[most_urgent_idx]
            # # 设计奖励函数（距离越近奖励越高，时间越紧迫权重越大）
            # urgency_weight = 1.0 - time_remaining[most_urgent_idx]  # 紧迫性系数[0,1]
            # a = distance(self.uav_pos[i], self.task_pos[most_urgent_task][:2])
            # # temp = (1.0 - a / MAP_SIZE)
            # dist_factor = np.exp(-self.xi * a)
            # # R_TW = urgency_weight * dist_factor #1.0 - time_remaining  #  # 时间窗奖励
            # R_TW = (1.0 - time_remaining[k]) if k in withinTW else 0

            # 增强版时间窗奖励（关键修改）
            time_remaining = remain_time(self.t, self.task_window[k])
            urgency = 1.0 / (time_remaining + 1e-5)  # 剩余时间越少，紧迫度越高

            # 距离衰减因子（严格单调递减）
            dist_factor = np.exp(-2.0 * dist / self.d_max)  # 更陡峭的衰减

            # 运动方向奖励（新增）
            prev_dist = distance(prev_pos[i], self.task_pos[k][:2])
            moving_toward = 1.2 if dist < prev_dist else 0.8  # 趋近奖励/远离惩罚

            R_TW = urgency * dist_factor * moving_toward

            # R_D = 0.5 * np.exp(-self.xi * a) + 0.5 / (1.0 + self.xi * a)
            # R_D = (0.5 * np.exp(-self.xi * dist) + 0.5 / (1.0 + self.xi * dist)) * R_TW   # 目标靠近 shaping 奖励
            # 边界惩罚
            R_border = -10.0 if (self.uav_pos[i][0] <= 0 or self.uav_pos[i][0] >= MAP_SIZE or
                                 self.uav_pos[i][1] <= 0 or self.uav_pos[i][1] >= MAP_SIZE) else 0.0

            # 使得能量和任务量成比例
            # R_fair = -abs((self.load[i] + 1e-5) / (np.sum(self.load) + 1e-5) -
            #               (self.energy[i] + 1e-5) / (np.sum(self.energy) + 1e-5))

            R_switch = 2.0 if k == most_urgent else 0.0

            # 惩罚原地滞留
            R_idle = -1.0 if v < 1e-3 else 0.0

            # 综合计算当前 UAV 的总奖励
            reward = (self.a1 * R_S + self.a2 * R_D - self.a3 * R_E / E_MAX - self.a4 * R_C  # + self.a6 * R_attract
                      + self.a7 * R_border + self.a8 * R_TW + 5.0 * R_switch)  # + self.a8 * R_fair + 2.0 * R_idle)
            rewards.append(reward)

            # 群体感知判定与任务完成,在群体任务完成判定后
        p_group = 0
        for j in unfinished:
            if len(task_participants[j]) > 0 and self.task_window[j][0] <= self.t <= self.task_window[j][1]:
                ps = [1 - p_individual[i, j] for i in task_participants[j]]
                p_group = 1 - np.prod(ps)
                p_group *= delta_k  # 时间窗进行调制
                if p_group >= self.p_th:
                    self.task_done[j] = p_group
                    self.task_marked[j] = True
                    self.finish_time[j] = self.t
                    # 平均奖励发放给参与者
                    for i in task_participants[j]:
                        rewards[i] += self.a5 * p_group / self.num_uav  # 降低主力 UAV 奖励，但可提升整体协作性

        # 绘制奖励函数曲线
        self.reward_history['R_S'].append(self.a1 * R_S)
        self.reward_history['R_D'].append(self.a2 * R_D)
        # self.reward_history['R_E'].append(R_E)
        self.reward_history['R_C'].append(self.a4 * R_C)
        self.reward_history['R_global'].append(self.a5 * p_group / self.num_uav)
        # self.reward_history['R_attract'].append(self.a6 * R_attract)
        self.reward_history['R_border'].append(self.a7 * R_border)
        self.reward_history['R_TW'].append(self.a8 * R_TW)

        info = {
            'tasks_completed': np.sum(self.task_marked) / self.num_task,
            'energy': self.energy.copy(),
            'collision': self._count_collisions(),
        }

        # 检查是否所有任务都已完成
        self.all_tasks_done = np.all(self.task_marked)
        # done = np.array([self.t >= self.T or all_tasks_done] * self.num_task, dtype=np.float32)#记录是否全部任务被完成或时间是否超过
        # done = np.array([self.task_done[k] for k in range(self.num_task)], dtype=np.float32)#独立记录每个任务是否被完成
        done = self.t > self.T or self.all_tasks_done

        # 所有任务都被完成
        if done:
            if self.all_tasks_done:
                info['all_tasks_done_time'] = self.t
            return self._get_obs(), np.array(rewards, dtype=np.float32), done, info
        else:
            self.t += 1
        return self._get_obs(), np.array(rewards, dtype=np.float32), done, info

    def _calculate_collision_reward_single(self, i):
        reward = 0.0
        for j in range(self.num_uav):
            if i != j:
                d_ij = distance(self.uav_pos[i], self.uav_pos[j])
                if d_ij < self.d_safe:
                    reward += 1.0
        return reward

    def _count_collisions(self):
        count = 0
        for i in range(self.num_uav):
            for j in range(i + 1, self.num_uav):
                d = distance(self.uav_pos[i], self.uav_pos[j])
                if d < self.d_safe:
                    count += 1
        return count
