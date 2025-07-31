import numpy as np
import random
from env import UAVEnv
from Plotter import Plotter
from config import TIME_STEPS, UAV_NUM, TASK_NUM, V_MAX
import math
from config import *
from datetime import datetime


class GA:
    def __init__(self,
                 time_window_type,
                 population_size=20,
                 generations=800,
                 mutation_rate=0.05,
                 elite_fraction=0.2):
        self.time_window_type = time_window_type
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.env = UAVEnv(time_window_type)  # 场景不变
        self.plotter = Plotter()
        self.best_chromosome = None
        self.best_reward = -np.inf
        self.best_trajectory = None
        self.completion_curve = []  # 任务完成率

    def initialize_population(self):
        """初始化种群，每个个体是一组时间步动作序列"""
        population = []
        for _ in range(self.population_size):
            individual = []
            for _ in range(TIME_STEPS):
                step_actions = []
                for _ in range(UAV_NUM):
                    v = np.random.uniform(0, V_MAX)
                    theta = np.random.uniform(-np.pi, np.pi)
                    task = np.random.randint(0, TASK_NUM)
                    step_actions.append((v, theta, task))
                individual.append(step_actions)
            population.append(individual)
        return population

    def evaluate_fitness(self, chromosome):
        """执行模拟环境并返回适应度"""
        env = UAVEnv(self.time_window_type)  # 每次新环境
        state = env.reset(self.time_window_type)
        total_reward = np.zeros(UAV_NUM)
        for t in range(TIME_STEPS):
            # unfinished_mask = np.array([k if self.env.task_done[k] < P_TH else 0.0 for k in range(TASK_NUM)])
            unfinished_task = set(np.where(self.env.task_done < P_TH)[0])
            # 修改当前步的动作，剔除已完成任务
            modified_actions = []
            for uav_action in chromosome[t]:
                v, theta, task = uav_action
                if task not in unfinished_task:
                    # 重新选择未完成的任务
                    if len(unfinished_task) > 0:
                        task = np.random.choice(unfinished_task)
                    else:
                        task = -1  # 所有任务已完成
                modified_actions.append((v, theta, task))

            # actions = chromosome[t]
            _, rewards, done, info = env.step(modified_actions)
            total_reward += rewards
            if done.all():
                break

        # 加权组合适应度
        path_length_penalty = sum([len(traj) for traj in env.uav_trajectories])#轨迹长度惩罚
        fitness = (
                np.mean(total_reward)
                + 100 * info['tasks_completed'] #100
                - 100 * info['collision']#50
                - 10 * path_length_penalty  #0.1 新增惩罚路径复杂性
        )

        return fitness, env

    def selection(self, population, fitnesses):
        """锦标赛选择两两个体对比"""
        selected = []
        for _ in range(len(population)):
            i, j = random.sample(range(len(population)), 2)
            selected.append(population[i] if fitnesses[i] > fitnesses[j] else population[j])
        return selected

    def crossover(self, parent1, parent2):
        """时间轴均分交叉"""
        cut = np.random.randint(1, TIME_STEPS - 1)
        child = parent1[:cut] + parent2[cut:]
        return child

    def mutate(self, chromosome):
        """以一定概率对动作进行扰动"""
        for t in range(TIME_STEPS):
            for i in range(UAV_NUM):
                if np.random.rand() < self.mutation_rate:
                    v = np.clip(chromosome[t][i][0] + np.random.randn(), 0, V_MAX)
                    theta = chromosome[t][i][1] + np.random.uniform(-0.1, 0.1)
                    # 只从未完成的任务中进行选择
                    unfinished_task = set(np.where(self.env.task_done < P_TH)[0])
                    if len(unfinished_task) > 0:
                        task = random.sample(unfinished_task, 1)[0]#随机选一个任务
                    else:
                        task = -1  # 所有任务已完成
                    # task = np.random.randint(0, TASK_NUM)
                    chromosome[t][i] = (v, theta, task)
        return chromosome

    def run(self):
        population = self.initialize_population()
        for gen in range(self.generations):
            fitnesses = []
            evaluated_envs = []
            for individual in population:
                fitness, sim_env = self.evaluate_fitness(individual)
                fitnesses.append(fitness)
                evaluated_envs.append(sim_env)

            # 保存最优
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_reward:
                self.best_reward = fitnesses[best_idx]
                self.best_chromosome = population[best_idx]
                self.best_trajectory = evaluated_envs[best_idx].uav_trajectories
                self.best_env = evaluated_envs[best_idx]
            # 记录任务完成率
            self.completion_curve.append(evaluated_envs[best_idx].task_done.mean())
            print(f"Generation {gen}, Best fitness: {fitnesses[best_idx]:.2f}, Task Completion: {evaluated_envs[best_idx].task_done.mean():.2f}")

            # 选择 + 精英保留
            elite_num = int(self.elite_fraction * self.population_size)
            sorted_indices = np.argsort(fitnesses)[::-1]
            elites = [population[i] for i in sorted_indices[:elite_num]]

            selected = self.selection(population, fitnesses)
            next_population = elites.copy()
            while len(next_population) < self.population_size:
                p1, p2 = random.sample(selected, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_population.append(child)
            population = next_population


        print("遗传算法完成.")
        return self.best_chromosome, self.completion_curve, datetime.now().strftime("%Y%m%d_%H%M%S")

    def visualize(self, timestamp):
        print("绘制最优轨迹与奖励曲线...")
        self.plotter.plot_uav_trajectories(self.best_trajectory, self.best_env.task_pos[:, :2],
                                           title="GA UAV Trajectories", save_name=f"GA_uav_trajectory_{timestamp}")
        self.plotter.plot_reward_components(self.best_env.reward_history)

    #统计任务完成率
    # def evaluate_task_completion(task_done, P_req):
    #     """
    #     task_done: 长度为 TASK_NUM 的 array，记录每个任务累计感知概率
    #     P_req: 每个任务的感知要求阈值（如 1.0）
    #     返回任务完成率
    #     """
    #     completed = np.sum(task_done >= P_req)
    #     total = len(task_done)
    #     return completed / total



