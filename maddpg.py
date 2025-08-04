import numpy as np
import tensorflow as tf
import random
from config import *
from collections import deque
import os
import time


class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = tf.keras.layers.Dense(400, kernel_initializer='he_uniform')
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.fc2 = tf.keras.layers.Dense(300, kernel_initializer='he_uniform')
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.swish = tf.keras.layers.Activation('swish')

        # 三个输出分支
        self.out_speed = tf.keras.layers.Dense(1, activation='sigmoid',
                                               kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3))
        self.out_angle = tf.keras.layers.Dense(1, activation='tanh',
                                               kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3))
        self.out_task = tf.keras.layers.Dense(TASK_NUM, activation='softmax')
        # 强制在 GPU 上初始化
        with tf.device('/GPU:0'):
            # self.build(input_shape=(None, obs_dim))  # 触发权重初始化在 GPU 上
            dummy_input = tf.random.normal((1, obs_dim))
            _ = self.call(dummy_input)  # 自动构建

    def call(self, inputs):
        x = self.swish(self.ln1(self.fc1(inputs)))
        x = self.swish(self.ln2(self.fc2(x)))
        speed = self.out_speed(x)
        angle = self.out_angle(x)
        task = self.out_task(x)
        res = tf.concat([speed, angle, task], axis=-1)
        return res


class Critic(tf.keras.Model):
    def __init__(self, obs_all_dim, act_all_dim):
        super(Critic, self).__init__()
        self.obs_fc = tf.keras.layers.Dense(400, kernel_initializer='he_uniform')
        self.obs_ln = tf.keras.layers.LayerNormalization()
        self.concat_ln = tf.keras.layers.LayerNormalization()
        self.concat_fc = tf.keras.layers.Dense(300, kernel_initializer='he_uniform')
        self.out_fc = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3))
        self.total_input_dim = 400 + act_all_dim  # 明确期望输入维度

        # 保存输入维度用于构建
        self.obs_all_dim = obs_all_dim
        self.act_all_dim = act_all_dim

        # 强制在 GPU 上初始化
        with tf.device('/GPU:0'):
            # self.build(input_shape=[(None, obs_all_dim), (None, act_all_dim)])
            # 使用虚拟数据自动构建
            dummy_obs = tf.random.normal((1, obs_all_dim))
            dummy_act = tf.random.normal((1, act_all_dim))
            _ = self.call(dummy_obs, dummy_act)  # 这将自动初始化权重

    def call(self, obs, act):
        obs = tf.cast(obs, tf.float32)
        act = tf.cast(act, tf.float32)
        x_obs = tf.nn.relu(self.obs_ln(self.obs_fc(obs)))
        x = tf.concat([x_obs, act], axis=-1)
        assert x.shape[
                   -1] == self.total_input_dim, f"[Critic] Input dim mismatch: got {x.shape[-1]}, expected {self.total_input_dim}"
        x = tf.nn.relu(self.concat_ln(self.concat_fc(x)))
        return self.out_fc(x)


class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer = deque(maxlen=buffer_size)
        self.num_agents = TASK_NUM  # 需要传入环境参数或智能体数量

    def add(self, transition):
        states, actions, rewards, next_states, done = transition
        # 将布尔done转换为每个智能体的done信号
        done_array = np.full(self.num_agents, done, dtype=np.float32)
        self.buffer.append((states, actions, rewards, next_states, done_array))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        # return tuple(tf.stack(tensors) for tensors in zip(*batch))  # 合并为批次张量
        # 在采样时转换为GPU张量
        with tf.device('/GPU:0'):
            # 确保所有张量转换为正确的形状
            obs = tf.convert_to_tensor(np.array([t[0] for t in batch]),
                                       dtype=tf.float32)  # (batch, num_agents, obs_dim)
            act = tf.convert_to_tensor(np.array([t[1] for t in batch]),
                                       dtype=tf.float32)  # (batch, num_agents, act_dim)
            rew = tf.convert_to_tensor(np.array([t[2] for t in batch]), dtype=tf.float32)  # (batch, num_agents)
            next_obs = tf.convert_to_tensor(np.array([t[3] for t in batch]),
                                            dtype=tf.float32)  # (batch, num_agents, obs_dim)
            # done = tf.convert_to_tensor(np.array([t[4] for t in batch]), dtype=tf.float32)  # (batch, num_agents)

        return obs, act, rew, next_obs

    def __len__(self):
        return len(self.buffer)


class MADDPG:
    def __init__(self, env):
        self.env = env
        self.num_agents = env.num_uav
        self.obs_dim = len(env._get_obs()[0])
        self.act_dim = 2 + TASK_NUM  # 不是 3，而是 [speed, angle, task_prob]
        self.train_step = 0
        self.actor = [Actor(self.obs_dim, self.act_dim) for _ in range(self.num_agents)]

        self.actor_optim = [tf.keras.optimizers.Adam(1e-4) for _ in range(self.num_agents)]  # 原来为1e-3
        self.critic_optim = [tf.keras.optimizers.Adam(1e-3) for _ in range(self.num_agents)]  # 原来为1e-3

        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.95
        self.batch_size = 256  # 256#64

        # 初始化目标网络
        self.target_actor = [Actor(self.obs_dim, self.act_dim) for _ in range(self.num_agents)]
        obs_all_dim = self.obs_dim * self.num_agents
        act_all_dim = self.act_dim * self.num_agents

        self.critic = [Critic(obs_all_dim, act_all_dim) for _ in range(self.num_agents)]
        self.target_critic = [Critic(obs_all_dim, act_all_dim) for _ in range(self.num_agents)]

        self.noise_std = 0.1  # 0.2  # 控制随机性幅度
        self.epsilon = 1.0  # 初始探索率为 1.0，逐步衰减
        self.epsilon_min = 0.1  # 0.05
        self.epsilon_decay = 0.997  # 0.995

        # 保存运行数据
        self.finish_time = {}
        self.best_task_rate = 0.0
        self.uav_trajectories = [[] for _ in range(UAV_NUM)]

        # 为智能体并行化预先分配内存
        self.obs_flat = tf.Variable(tf.zeros([self.batch_size, self.obs_dim * self.num_agents]))
        self.next_obs_flat = tf.Variable(tf.zeros([self.batch_size, self.obs_dim * self.num_agents]))
        self.act_flat = tf.Variable(tf.zeros([self.batch_size, self.act_dim * self.num_agents]))
        self.target_acts_flat = tf.Variable(tf.zeros([self.batch_size, self.act_dim * self.num_agents]))
        self.new_acts_flat = tf.Variable(tf.zeros([self.batch_size, self.act_dim * self.num_agents]))

        # 拷贝权重
        for i in range(self.num_agents):
            self.update_target_network(self.target_actor[i], self.actor[i])
            self.update_target_network(self.target_critic[i], self.critic[i])

        # 保存模型Checkpointing
        self.ckpt_dir = "MADDPG_checkpoints"
        self.actor_ckpt = os.path.join(self.ckpt_dir, "actor")
        self.critic_ckpt = os.path.join(self.ckpt_dir, "critic")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 初始化所有网络到 GPU
        with tf.device('/GPU:0'):
            self.actor = [Actor(self.obs_dim, self.act_dim) for _ in range(self.num_agents)]
            self.target_actor = [Actor(self.obs_dim, self.act_dim) for _ in range(self.num_agents)]
            self.critic = []
            self.target_critic = []
            with tf.device('/GPU:0'):
                for _ in range(self.num_agents):
                    # 创建实例后立即用虚拟数据调用以初始化权重
                    critic = Critic(obs_all_dim, act_all_dim)
                    target_critic = Critic(obs_all_dim, act_all_dim)

                    # 确保权重初始化
                    dummy_obs = tf.random.normal((1, obs_all_dim))
                    dummy_act = tf.random.normal((1, act_all_dim))
                    _ = critic(dummy_obs, dummy_act)
                    _ = target_critic(dummy_obs, dummy_act)

                    self.critic.append(critic)
                    self.target_critic.append(target_critic)

    def save_models(self):
        """保存所有智能体的模型到指定目录"""
        # 确保目录存在
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 保存所有Actor和Critic
        for i in range(self.num_agents):
            actor_path = os.path.join(self.ckpt_dir, f"actor_{i}.h5")
            critic_path = os.path.join(self.ckpt_dir, f"critic_{i}.h5")
            self.actor[i].save_weights(actor_path)
            self.critic[i].save_weights(critic_path)

        print("Models saved!")

    def load_models(self, epoch=None):
        """从指定目录加载所有智能体的模型"""
        # 检查目录是否存在
        if not os.path.exists(self.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory {self.ckpt_dir} not found")

        for i in range(self.num_agents):
            actor_path = os.path.join(self.ckpt_dir, f"actor_{i}")
            critic_path = os.path.join(self.ckpt_dir, f"critic_{i}")
            #检查文件是否存在
            if not (os.path.exists(actor_path + ".index") and os.path.exists(critic_path + ".index")):
                raise FileNotFoundError(f"Model checkpoints for agent {i} not found")
            self.actor[i].load_weights(actor_path)
            self.critic[i].load_weights(critic_path)


        # 先进行虚拟前向传播以构建变量
        # dummy_obs = tf.random.normal((1, self.obs_dim * self.num_agents))
        # dummy_act = tf.random.normal((1, self.act_dim * self.num_agents))
        # # 加载所有Actor和Critic
        # for i in range(self.num_agents):
        #     actor_path = os.path.join(self.ckpt_dir, f"actor_{i}.h5")
        #     critic_path = os.path.join(self.ckpt_dir, f"critic_{i}.h5")
        # 构建Actor变量
        #     _ = self.actor[i](np.zeros((1, self.obs_dim)))  # 关键步骤：触发变量创建
        #     self.actor[i].load_weights(actor_path)
        #
        #     # 构建Critic变量
        #     _ = self.critic[i](dummy_obs, dummy_act)  # 关键步骤
        #     self.critic[i].load_weights(critic_path)
        print("Models loaded!")

    def act(self, obs):
        actions_for_env = []
        actions_for_train = []
        # 添加输入检查
        if np.any(np.isnan(obs)):
            print("Warning: NaN in observation!")
            obs = np.nan_to_num(obs, nan=0.0)
        for i in range(self.num_agents):
            with tf.device('/GPU:0'):
                # 直接切片获取单个智能体的观测（避免重复转换）
                obs_i = obs[i:i + 1]  # 保持批处理维度 [1, obs_dim]
                output = self.actor[i](obs_i)
                output = output[0].numpy()  # 取第一个（也是唯一一个）样本

            # 加噪探索
            if np.random.rand() < self.epsilon:
                speed = np.random.uniform(0, 1) * V_MAX
                angle = np.random.uniform(-1, 1) * np.pi
                task_id = np.random.choice(TASK_NUM)
                task_softmax = np.zeros(TASK_NUM)
                task_softmax[task_id] = 1.0  # one-hot 用于训练
            else:
                speed_raw = output[0] + np.random.normal(0, self.noise_std)
                angle_raw = output[1] + np.random.normal(0, self.noise_std)
                task_probs = np.clip(output[2:], 1e-8, 1.0)

                unfinished_task = np.array([1.0 if self.env.task_done[k] < P_TH else 0.0 for k in range(TASK_NUM)])
                task_probs *= unfinished_task
                if task_probs.sum() > 0:
                    task_probs /= task_probs.sum()
                else:
                    task_probs = np.ones(TASK_NUM) / TASK_NUM
                task_id = np.random.choice(TASK_NUM, p=task_probs)

                task_softmax = task_probs  # 用于训练

                speed = np.clip(speed_raw, 0, 1) * V_MAX
                angle = np.clip(angle_raw, -1, 1) * np.pi

            # 给环境的动作（执行）
            actions_for_env.append([speed, angle, task_id])

            # 用于训练的动作（带 softmax）
            train_action = np.concatenate([
                [speed / V_MAX],  # 归一化 speed
                [angle / np.pi],  # 归一化 angle
                task_softmax  # softmax
            ])
            actions_for_train.append(train_action)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return actions_for_env, actions_for_train

    def learn(self):
        # 从回放缓存中采样一个批次
        obs_batch, act_batch, reward_batch, next_obs_batch = self.replay_buffer.sample(self.batch_size)

        # 确保维度合法
        assert obs_batch.shape == (self.batch_size, self.num_agents, self.obs_dim)
        assert act_batch.shape == (self.batch_size, self.num_agents, self.act_dim)
        assert reward_batch.shape == (self.batch_size, self.num_agents)

        # 提前 reshape 和 concat
        obs_concat = tf.reshape(obs_batch, [self.batch_size, -1])  # (batch, num_agents*obs_dim)
        act_concat = tf.reshape(act_batch, [self.batch_size, -1])  # (batch, num_agents*act_dim)
        next_obs_concat = tf.reshape(next_obs_batch, [self.batch_size, -1])

        with tf.device('/GPU:0'):
            for i in range(self.num_agents):  # 遍历每个智能体
                # === Critic 更新 ===
                with tf.GradientTape() as tape:  # 使用梯度带记录 Critic 网络的梯度
                    # 计算目标动作（所有智能体）
                    target_acts = [
                        self.target_actor[j](next_obs_batch[:, j])
                        for j in range(self.num_agents)
                    ]
                    target_acts_concat = tf.concat(target_acts, axis=-1)  # concat沿指定轴将多个frame对象拼接在一起

                    # 计算目标Q值
                    target_q = self.target_critic[i](next_obs_concat, target_acts_concat)  # 通过目标 Critic 网络计算目标 Q 值
                    y = reward_batch[:, i] + self.gamma * tf.squeeze(target_q)  # 直接使用 gamma 乘 target_q
                    y = tf.clip_by_value(y, -1000.0, 100.0)  # 保持值裁剪

                    current_q = self.critic[i](obs_concat, act_concat)
                    critic_loss = tf.keras.losses.Huber()(y, tf.squeeze(current_q))

                # 更新 Critic
                critic_grads = tape.gradient(critic_loss, self.critic[i].trainable_variables)  # 计算 Critic 的梯度
                if any(np.any(np.isnan(g.numpy())) for g in critic_grads):
                    print(f"NaN gradients in critic {i}!")
                self.critic_optim[i].apply_gradients(
                    zip(critic_grads, self.critic[i].trainable_variables))  # 应用梯度更新 Critic 网络

                # === Actor 更新 ===
                with tf.GradientTape() as tape:
                    # 生成新动作（所有智能体）
                    new_actions = [
                        self.actor[j](obs_batch[:, j])
                        for j in range(self.num_agents)
                    ]
                    actor_loss = -tf.reduce_mean(self.critic[i](
                        tf.reshape(obs_batch, [self.batch_size, -1]),
                        tf.concat(new_actions, axis=-1)
                    ))  # Actor 的目标是最大化 Q 值，因此损失函数是 Critic 输出的负值

                # 更新 Actor
                actor_grads = tape.gradient(actor_loss, self.actor[i].trainable_variables)  # 计算 Actor 的梯度
                if any(np.any(np.isnan(g.numpy())) for g in actor_grads):
                    print(f"NaN gradients in actor {i}!")
                self.actor_optim[i].apply_gradients(
                    zip(actor_grads, self.actor[i].trainable_variables))  # 应用梯度更新 Actor 网络

        # 目标网络更新（每隔10步）
        if self.train_step % 10 == 0:
            for i in range(self.num_agents):
                self.update_target_network(self.target_actor[i], self.actor[i], tau=0.01)  # 更新目标 Actor 网络
                self.update_target_network(self.target_critic[i], self.critic[i], tau=0.01)  # 更新目标 Critic 网络
        self.train_step += 1

    def update_target_network(self, target, source, tau=0.01):  # tau=1.0 为硬拷贝，可改软更新;0.01为软更新
        for target_param, param in zip(target.trainable_variables, source.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

    # 训练
    def train(self, epochs=500):

        # 多次运行以获取统计数据
        all_rewards = []
        all_task_completion = []
        all_collisions = []
        all_fairness = []

        for episode in range(epochs):  # 100
            # 记录开始时间
            start_time = time.time()

            states = self.env.reset(self.env.time_window_type)
            episode_reward = np.zeros(self.num_agents)

            for t in range(TIME_STEPS):

                # 将状态数据转换为 GPU 张量
                with tf.device('/GPU:0'):
                    states_gpu = tf.convert_to_tensor(states, dtype=tf.float32)

                # 在 GPU 上执行动作预测
                with tf.device('/GPU:0'):
                    actions, actions_train = self.act(states_gpu)  # 获取动作

                # next_states, rewards, done, info = self.env.step(actions)  # 执行动作
                try:
                    next_states, rewards, done = self.env.step(actions)
                except Exception as e:
                    print(f"Env step failed: {str(e)}")
                    break

                if done:
                    if self.env.all_tasks_done_time >= 0:
                        print(f"所有任务完成！耗时步数：{self.env.t}")
                    else:
                        print(f"没有可以完成的任务！")
                    break

                # 直接传入原始数据（无需手动转换到GPU）
                self.replay_buffer.add((
                    states,  # 旧视野
                    actions_train,  # 走法
                    rewards,
                    next_states,  # 新视野
                    done  # 原始终止标志
                ))

                # 更新状态和奖励
                states = next_states
                episode_reward += rewards  # 添加元素

                # 学习
                if t % 10 == 0 and len(self.replay_buffer) > self.batch_size:
                    with tf.device('/GPU:0'):
                        self.learn()

            # 记录本轮的统计数据
            all_rewards.append(np.sum(episode_reward))
            tasks_completed = self.env.tasks_completed/TASK_NUM
            all_task_completion.append(tasks_completed)
            collision = self.env.count_collisions
            all_collisions.append(collision)
            fairness = self.env.calculate_fairness()
            all_fairness.append(fairness)

            print(f"回合 {episode}, 平均奖励: {np.mean(episode_reward):.2f}, "
                  f"任务完成率: {tasks_completed:.2f}, 碰撞次数: {collision}, 能耗均衡: {fairness}")

            # if episode < epochs/2:
            #     self.uav_trajectories = self.env.uav_trajectories
            # el
            if self.best_task_rate <= tasks_completed:
                self.best_task_rate = tasks_completed
                self.uav_trajectories = self.env.uav_trajectories
                self.finish_time = self.env.finish_time

            # 记录本次运行的所有回合数据
            # 记录结束时间
            end_time = time.time()

            # 计算间隔时间
            interval = end_time - start_time

            # 打印间隔时间
            print(f"运行 {episode} 完成，耗时 {interval:.2f} 秒")

        self.save_models()
        return all_rewards, all_task_completion, all_collisions, all_fairness

    def test(self, epochs=500):
        self.epsilon = 0.0  # 禁用探索
        self.load_models()
        all_rewards = []
        all_task_completion = []
        all_collisions = []
        all_fairness = []

        for episode in range(epochs):  # 100
            # 记录开始时间
            start_time = time.time()

            states = self.env.reset(self.env.time_window_type)
            episode_reward = np.zeros(self.num_agents)

            for t in range(TIME_STEPS):

                # 将状态数据转换为 GPU 张量
                with tf.device('/GPU:0'):
                    states_gpu = tf.convert_to_tensor(states, dtype=tf.float32)

                # 在 GPU 上执行动作预测
                with tf.device('/GPU:0'):
                    actions, actions_train = self.act(states_gpu)  # 获取动作

                # next_states, rewards, done, info = self.env.step(actions)  # 执行动作
                try:
                    next_states, rewards, done, info = self.env.step(actions)
                except Exception as e:
                    print(f"Env step failed: {str(e)}")
                    break

                # 如果所有任务完成
                if "all_tasks_done_time" in info:
                    break  # 退出当前 episode

                # 更新状态和奖励
                states = next_states
                episode_reward += rewards
                # 学习
                if t % 10 == 0 and len(self.replay_buffer) > self.batch_size:
                    with tf.device('/GPU:0'):
                        self.learn()

            # 记录本轮的统计数据
            all_rewards.append(np.sum(episode_reward))
            all_task_completion.append(info['tasks_completed'])
            all_collisions.append(info['collision'])
            all_fairness.append(info['fairness'])

            print(f"回合 {episode}, 平均奖励: {np.mean(episode_reward):.2f}, "
                  f"任务完成率: {info['tasks_completed']:.2f}, 碰撞次数: {info['collision']}")

            # if episode < epochs/2:
            #     self.uav_trajectories = self.env.uav_trajectories
            # el
            if self.best_task_rate <= info['tasks_completed']:
                self.best_task_rate = info['tasks_completed']
                self.uav_trajectories = self.env.uav_trajectories
                self.finish_time = self.env.finish_time

            # 记录本次运行的所有回合数据
            # 记录结束时间
            end_time = time.time()

            # 计算间隔时间
            interval = end_time - start_time

            # 打印间隔时间
            print(f"运行 {episode} 完成，耗时 {interval:.2f} 秒")

        return all_rewards, all_task_completion, all_collisions, all_fairness
