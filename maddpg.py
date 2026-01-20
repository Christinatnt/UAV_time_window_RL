import numpy as np
import tensorflow as tf
import random
from config import *
from collections import deque
import os
import time
from utils import *


class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim=None):
        super(Actor, self).__init__()
        self.fc1 = tf.keras.layers.Dense(400, activation='swish', kernel_initializer='he_uniform')
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.fc2 = tf.keras.layers.Dense(300, activation='swish', kernel_initializer='he_uniform')
        self.ln2 = tf.keras.layers.LayerNormalization()

        # 只学任务（高层决策）
        self.out_task = tf.keras.layers.Dense(TASK_NUM, activation='softmax',
                                              kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3))
        # 可选：小残差，用于在 RL 阶段对控制器做微调（BC/DAgger不监督）
        self.out_dv = tf.keras.layers.Dense(1, activation='tanh',
                                            kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3))
        self.out_dth = tf.keras.layers.Dense(1, activation='tanh',
                                             kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3))

        with tf.device('/GPU:0'):
            dummy_input = tf.random.normal((1, obs_dim))
            _ = self.call(dummy_input)

    def call(self, inputs):
        x = self.ln1(self.fc1(inputs))
        x = self.ln2(self.fc2(x))
        p_task = self.out_task(x)  # [B, TASK_NUM]
        dv = self.out_dv(x)  # [B, 1]   in [-1, 1]
        dth = self.out_dth(x)  # [B, 1]   in [-1, 1]
        return p_task, dv, dth


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
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995  # 0.995

        self.use_residual = False  # 先关残差，稳定到>85%后再开
        self.res_eps_v = 0.1  # 开残差时建议 0.05~0.2
        self.res_eps_th = 0.1
        self.lambda_im = 0.0  # 纯RL阶段 =0；DAgger阶段>0（如 0.5 ~ 2.0）
        # 单独存 DAgger/BC 的任务监督数据（只存任务 onehot，不存速度/角度）
        self.dagger_buf = [deque(maxlen=200000) for _ in range(self.num_agents)]

        # 保存运行数据
        self.finish_time = {}
        self.best_task_rate = 0.0
        self.uav_trajectories = [[] for _ in range(UAV_NUM)]
        self.task_marked = np.zeros(UAV_NUM, dtype=bool)  # 标记任务是否完成Bool
        self.loss = []

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

        # 添加速度角度平滑
        self.alpha_th = 0.5  # 角度平滑(0.1-0.3)
        self.beta_v = 0.8  # 速度平滑
        self.prev_v = [0.0] * self.num_agents
        self.prev_th = [0.0] * self.num_agents

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
            # 检查文件是否存在
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

    # 低层控制器
    def _controller_to_task(self, uav_pos, task_xy):
        vec = task_xy - uav_pos
        dist = np.linalg.norm(vec)
        base_th = np.arctan2(vec[1], vec[0])
        # 也可以用 dist/τ 平滑速度，这里先用最简单的剪裁
        base_v = np.clip(dist, 0, V_MAX)
        return base_v, base_th

    def act(self, obs):
        actions_for_env = []
        actions_for_train = []

        if isinstance(obs, tf.Tensor):
            obs_np = obs.numpy()
        else:
            obs_np = np.asarray(obs, dtype=np.float32)

        for i in range(self.num_agents):
            with tf.device('/GPU:0'):
                p_task, dv_raw, dth_raw = self.actor[i](obs_np[i:i + 1])
                p_task = p_task[0].numpy()  # [TASK_NUM]
                dv_raw = float(dv_raw[0, 0].numpy())
                dth_raw = float(dth_raw[0, 0].numpy())

            # mask 已完成/超时任务（防止选到无效任务）
            unfinished = np.array(
                [1.0 if self.env.task_done[k] < P_TH and self.env.t <= self.env.task_window[k][1] else 0.0
                 for k in range(TASK_NUM)], dtype=np.float32)
            p_task = np.clip(p_task, 1e-8, 1.0) * unfinished
            if p_task.sum() == 0:
                p_task = np.ones_like(p_task) / TASK_NUM
            else:
                p_task = p_task / p_task.sum()

            # 任务选择：可 epsilon-greedy，也可直接按分布采样
            if np.random.rand() < self.epsilon:
                task_id = np.random.choice(TASK_NUM, p=p_task)
            else:
                task_id = np.random.choice(TASK_NUM, p=p_task)

            # 低层控制器
            base_v, base_th = self._controller_to_task(self.env.uav_pos[i], self.env.task_pos[task_id][:2])

            # 可选：小残差（纯 RL 中起作用；BC/DAgger 不监督残差参数）
            if self.use_residual:
                v = np.clip(base_v + self.res_eps_v * dv_raw * V_MAX, 0, V_MAX)
                th = np.clip(base_th + self.res_eps_th * dth_raw * np.pi, -np.pi, np.pi)
            else:
                v, th = base_v, base_th

            # ========= 一阶平滑（关键） =========
            if self.train_step == 0 or self.train_step == 1:  # 第一步，避免使用未初始化的 prev
                v_smooth = v
                th_smooth = th
            else:
                v_smooth = (1 - self.beta_v) * self.prev_v[i] + self.beta_v * v

                # 角度要注意 wrap 到 [-pi, pi]
                th_diff = th - self.prev_th[i]
                th_diff = (th_diff + np.pi) % (2 * np.pi) - np.pi
                th_smooth = self.prev_th[i] + self.alpha_th * th_diff

            # 保存
            self.prev_v[i] = v_smooth
            self.prev_th[i] = th_smooth

            # 环境动作（执行）
            actions_for_env.append([v_smooth, th_smooth, task_id])

            # 训练动作（给 Critic 的连续向量）
            onehot = np.zeros(TASK_NUM, dtype=np.float32);
            onehot[task_id] = 1.0
            train_action = np.concatenate([[v / V_MAX], [th / np.pi], onehot], axis=0)
            actions_for_train.append(train_action)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return actions_for_env, actions_for_train

    def learn(self):
        # 从回放缓存中采样一个批次
        obs_batch, act_batch, reward_batch, next_obs_batch = self.replay_buffer.sample(self.batch_size)

        assert obs_batch.shape == (self.batch_size, self.num_agents, self.obs_dim)
        assert act_batch.shape == (self.batch_size, self.num_agents, self.act_dim)
        assert reward_batch.shape == (self.batch_size, self.num_agents)

        # reshape for critic
        obs_concat = tf.reshape(obs_batch, [self.batch_size, -1])
        act_concat = tf.reshape(act_batch, [self.batch_size, -1])
        next_obs_concat = tf.reshape(next_obs_batch, [self.batch_size, -1])

        ce_loss = tf.keras.losses.CategoricalCrossentropy()

        with tf.device('/GPU:0'):
            for i in range(self.num_agents):
                # === Critic 更新 ===
                with tf.GradientTape() as tape:
                    # target 动作（由 target_actor 生成）
                    target_actions = []
                    for j in range(self.num_agents):
                        p_task_j, dv, dth = self.target_actor[j](next_obs_batch[:, j])
                        p_task_soft = p_task_j  # shape [batch, TASK_NUM], 已经是 softmax
                        v_norm = dv / V_MAX
                        th_norm = dth / np.pi
                        target_actions.append(tf.concat([v_norm, th_norm, p_task_soft], axis=-1))

                    target_acts_concat = tf.concat(target_actions, axis=-1)

                    target_q = self.target_critic[i](next_obs_concat, target_acts_concat)
                    y = reward_batch[:, i] + self.gamma * tf.squeeze(target_q)
                    y = tf.clip_by_value(y, -1000.0, 100.0)

                    current_q = self.critic[i](obs_concat, act_concat)
                    critic_loss = tf.keras.losses.Huber()(y, tf.squeeze(current_q))

                critic_grads = tape.gradient(critic_loss, self.critic[i].trainable_variables)
                self.critic_optim[i].apply_gradients(zip(critic_grads, self.critic[i].trainable_variables))

                # === Actor 更新 ===
                with tf.GradientTape() as tape:
                    # 生成 actor 当前动作
                    new_actions = []
                    for j in range(self.num_agents):
                        p_task, dv, dth = self.actor[j](obs_batch[:, j])
                        v_norm = dv / V_MAX
                        th_norm = dth / np.pi
                        task_idx = tf.argmax(p_task, axis=-1)
                        onehot = tf.one_hot(task_idx, TASK_NUM)
                        new_actions.append(tf.concat([v_norm, th_norm, onehot], axis=-1))
                    new_act_concat = tf.concat(new_actions, axis=-1)

                    # Critic 评估新动作
                    q_val = self.critic[i](obs_concat, new_act_concat)
                    actor_loss = -tf.reduce_mean(q_val)

                    total_loss = actor_loss

                    # imitation loss（只在 BC/DAgger 阶段启用）
                    if self.lambda_im > 0 and len(self.dagger_buf[i]) >= self.batch_size:
                        idx = np.random.choice(len(self.dagger_buf[i]), size=self.batch_size, replace=False)
                        obs_im = tf.convert_to_tensor(np.array([self.dagger_buf[i][k][0] for k in idx]),
                                                      dtype=tf.float32)
                        task_im = tf.convert_to_tensor(np.array([self.dagger_buf[i][k][1] for k in idx]),
                                                       dtype=tf.float32)
                        p_task_im, _, _ = self.actor[i](obs_im)
                        imitation_loss = ce_loss(task_im, p_task_im)
                        total_loss += self.lambda_im * imitation_loss

                actor_grads = tape.gradient(total_loss, self.actor[i].trainable_variables)
                self.actor_optim[i].apply_gradients(zip(actor_grads, self.actor[i].trainable_variables))

            # === 软更新 target 网络 ===
            if self.train_step % 10 == 0:
                for i in range(self.num_agents):
                    self.update_target_network(self.target_actor[i], self.actor[i], tau=0.01)
                    self.update_target_network(self.target_critic[i], self.critic[i], tau=0.01)
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

        # 打开残差
        self.use_residual = True
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
            tasks_completed = self.env.task_marked.sum() / TASK_NUM
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
                self.task_marked = self.env.task_marked
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

    # 启发式
    def run_heuristic(self, epochs=100):
        """
        使用 heuristic 策略跑完整环境，用于公平对比
        """

        all_rewards = []
        all_task_completion = []
        all_collisions = []
        all_fairness = []

        for episode in range(epochs):
            start_time = time.time()

            states = self.env.reset(self.env.time_window_type)
            episode_reward = np.zeros(self.num_agents)

            for t in range(TIME_STEPS):

                # ===== 核心区别：用 heuristic 选动作 =====
                actions = []
                for i in range(self.num_agents):
                    action_i = self.heuristic_policy(i, states, self.env)
                    actions.append(action_i)
                actions = np.array(actions)

                try:
                    next_states, rewards, done = self.env.step(actions)
                except Exception as e:
                    print(f"Heuristic env step failed: {str(e)}")
                    break
                if done:
                    if self.env.all_tasks_done_time >= 0:
                        print(f"所有任务完成！耗时步数：{self.env.t}")
                    else:
                        print(f"没有可以完成的任务！")
                    break

                states = next_states
                episode_reward += rewards

                if done:
                    if self.env.all_tasks_done_time >= 0:
                        print(f"[Heuristic] 所有任务完成，步数={self.env.t}")
                    else:
                        print(f"[Heuristic] 无可完成任务")
                    break

            # ===== 与 MADDPG 完全一致的统计方式 =====

            # tasks_completed = self.env.task_marked.sum() / TASK_NUM
            # all_task_completion.append(tasks_completed+random.uniform(-0.02, 0.05))

            # collision = self.env.count_collisions
            # all_collisions.append(collision)
            #
            # fairness = self.env.calculate_fairness()
            # all_fairness.append(fairness)

            # 记录本轮的统计数据
            all_rewards.append(np.sum(episode_reward))
            tasks_completed = self.env.task_marked.sum() / TASK_NUM
            all_task_completion.append(tasks_completed)
            collision = self.env.count_collisions
            all_collisions.append(collision)
            fairness = self.env.calculate_fairness()
            all_fairness.append(fairness)

            print(f"[Heuristic] 回合 {episode}, 平均奖励: {np.mean(episode_reward):.2f}, "
                  f"任务完成率: {tasks_completed:.2f}, 碰撞次数: {collision}, 能耗均衡: {fairness}")

            if self.best_task_rate <= tasks_completed:
                self.best_task_rate = tasks_completed
                self.uav_trajectories = self.env.uav_trajectories
                self.task_marked = self.env.task_marked
                self.finish_time = self.env.finish_time

            end_time = time.time()
            print(f"[Heuristic] 运行 {episode} 完成，耗时 {end_time - start_time:.2f}s")

        return all_rewards, all_task_completion, all_collisions, all_fairness

    def heuristic_policy(self, agent_id, obs, env):
        """
        返回一个完整动作 [v, theta, task_onehot]
        """
        # 1) 选任务（你已有的专家逻辑）
        task_id = self.get_expert_action(agent_id, env, env.t)

        task_onehot = np.zeros(TASK_NUM, dtype=np.float32)
        task_onehot[task_id] = 1.0

        # 2) 控制（直接飞向目标）
        uav_pos = env.uav_pos[agent_id]
        task_pos = env.task_pos[task_id]

        speed, angle = self._controller_to_task(uav_pos, task_pos[:2])
        if self.train_step == 0 or self.train_step == 1:  # 第一步，避免使用未初始化的 prev
            v_smooth = speed
            th_smooth = angle
        else:
            v_smooth = (1 - self.beta_v) * self.prev_v[agent_id] + self.beta_v * speed
            th_diff = angle - self.prev_th[agent_id]
            th_diff = (th_diff + np.pi) % (2 * np.pi) - np.pi
            th_smooth = self.prev_th[agent_id] + self.alpha_th * th_diff

        return [v_smooth, th_smooth, task_id]

    # 专家策略生成
    def get_expert_action(self, uav_index, env, current_time):
        unfinished = [j for j in range(len(env.task_pos)) if not env.task_marked[j]]
        withinTW = [j for j in unfinished if current_time <= env.task_window[j][1]]
        if not withinTW:
            return -1

        scores = []
        for j in withinTW:
            urgency = 1.0 - remain_time(current_time, env.task_window[j])
            closest = distance(env.uav_pos[uav_index], env.task_pos[j][:2])
            scores.append(urgency / (closest + 1e-5))

        optimal_task = withinTW[np.argmax(scores)]
        return optimal_task

    # 数据生成器，用专家策略采集demo
    def collect_expert_data(self, env, time_window_type, num_episodes=1000):
        D_demo = []  # [(obs_i, task_onehot)]
        for _ in range(num_episodes):
            obs = env.reset(time_window_type)
            for t in range(TIME_STEPS):
                # 仅用于产生标签，不必真的用专家执行
                for i in range(env.num_uav):
                    task_id = self.get_expert_action(i, env, t)
                    if task_id < 0:  # 已经没有可以执行的任务了则跳出循环
                        break
                    onehot = np.zeros(TASK_NUM, dtype=np.float32)
                    onehot[task_id] = 1.0
                    D_demo.append((np.array(obs[i], dtype=np.float32), onehot))
                    # 存入 DAgger 监督池（只 obs_i, onehot）
                    self.dagger_buf[i].append((np.array(obs[i], dtype=np.float32), onehot))
                # 让环境以“控制器+当前actor任务”推进（也可以随机走几步）
                actions_env, _ = self.act(tf.convert_to_tensor(obs, dtype=tf.float32))
                obs, _, done = env.step(actions_env)
                if done: break
        return D_demo

    # 模仿学习：训练Actor网络是其输出模仿专家动作
    def train_behavior_cloning(self, D_demo, epochs=100):
        optimizers = [tf.keras.optimizers.Adam(5e-4) for _ in range(self.num_agents)]
        ce_loss = tf.keras.losses.CategoricalCrossentropy()

        # 先处理 D_demo 中第一个样本，将 obs 和 act 转为 numpy 数组获取形状
        # 假设 D_demo 元素是 (obs_list, act_list) 形式，这里取出第一个 obs 和 act 转成数组
        first_obs_np = np.array(D_demo[0][0], dtype=np.float32)
        first_act_np = np.array(D_demo[0][1], dtype=np.float32)

        # 直接从列表创建数据集，避免中间NumPy数组（修改 output_signature ）
        dataset = tf.data.Dataset.from_generator(
            lambda: ((obs, act) for obs, act in D_demo),
            output_signature=(
                tf.TensorSpec(shape=first_obs_np.shape, dtype=tf.float32),
                tf.TensorSpec(shape=first_act_np.shape, dtype=tf.float32)
            )
        )

        # 使用GPU（如果可用）进行后续处理
        with tf.device('/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'):
            dataset = dataset.shuffle(10000).batch(self.batch_size)

        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            # 记录开始时间
            start_time = time.time()
            for obs_batch, task_batch in dataset:
                batch_loss = 0
                for i, (actor, optimizer) in enumerate(zip(self.actor, optimizers)):  # 对每个actor都训练
                    with tf.device('/GPU:0'):
                        with tf.GradientTape() as tape:
                            p_task, _, _ = actor(obs_batch)  # 只用任务分支
                            loss = ce_loss(task_batch, p_task)

                        grads = tape.gradient(loss, actor.trainable_variables)
                        optimizer.apply_gradients(zip(grads, actor.trainable_variables))
                        epoch_loss += float(loss.numpy())
                n += 1
            avg_loss = epoch_loss / (n * self.num_agents)  # 平均每个batch的损失
            print(f"[BC] epoch {epoch} loss={avg_loss:.4f}")
            self.loss.append(avg_loss)  # 存储平均损失
            # 记录本次运行的所有回合数据
            # 记录结束时间
            end_time = time.time()

            # 计算间隔时间
            interval = end_time - start_time

            # 打印间隔时间
            print(f"运行 {epoch} 完成，耗时 {interval:.2f} 秒")

    def dagger_training(self, iterations=10, steps_per_iter=TIME_STEPS, lambda_im=1.0):
        self.lambda_im = float(lambda_im)  # 开启 imitation loss
        for it in range(iterations):
            print(f"[DAgger] Iteration {it + 1}")
            obs = self.env.reset(self.env.time_window_type)
            for t in range(steps_per_iter):
                actions_env, actions_train = self.act(tf.convert_to_tensor(obs, dtype=tf.float32))
                new_obs, reward, done = self.env.step(actions_env)

                # 专家只给任务标签
                actions_task_onehot = []
                for i in range(self.num_agents):
                    task_id = self.get_expert_action(i, self.env, self.env.t)
                    onehot = np.zeros(TASK_NUM, dtype=np.float32);
                    onehot[task_id] = 1.0
                    actions_task_onehot.append(onehot)
                    # 存入 DAgger 监督池（只 obs_i, onehot）
                    self.dagger_buf[i].append((np.array(obs[i], dtype=np.float32), onehot))

                # 仍把“真实执行动作=控制器(+残差)”存进 RL 回放
                self.replay_buffer.add((obs, actions_train, reward, new_obs, done))

                obs = new_obs
                if done: break

            # 每轮迭代后，做几轮 RL 学习（含 imitation-loss）
            for _ in range(20):
                if len(self.replay_buffer) > self.batch_size:
                    self.learn()

        self.lambda_im = 0.0  # 结束 DAgger，关闭 imitation-loss
