# mappo.py
import numpy as np
import tensorflow as tf
from config import *
import time
import os
# 明确使用 TF 内部的 keras 路径（IDE 更容易识别）
from tensorflow.python.keras.layers import Layer  # 具体到需要的层


# from tensorflow.keras import layers

class Actor(tf.keras.Model):
    def __init__(self, obs_dim):
        super().__init__()
        with tf.device('/GPU:0'):
            self.fc1 = tf.keras.layers.Dense(128, activation='tanh')
            self.fc2 = tf.keras.layers.Dense(128, activation='tanh')
            self.out_v = tf.keras.layers.Dense(1, activation='sigmoid')  # v ∈ [0, 1]
            self.out_theta = tf.keras.layers.Dense(1, activation='tanh')  # θ ∈ [-1, 1]
            self.out_task = tf.keras.layers.Dense(TASK_NUM, activation='softmax')  # task_id softmax

    def call(self, x):
        with tf.device('/GPU:0'):
            x = self.fc1(x)
            x = self.fc2(x)
            v = self.out_v(x)
            theta = self.out_theta(x)
            task = self.out_task(x)
            return v, theta, task


class Critic(tf.keras.Model):
    def __init__(self, obs_dim):
        super().__init__()
        with tf.device('/GPU:0'):
            self.fc1 = tf.keras.layers.Dense(128, activation='tanh')
            self.fc2 = tf.keras.layers.Dense(128, activation='tanh')
            self.out_v = tf.keras.layers.Dense(1)

    def call(self, x):
        with tf.device('/GPU:0'):
            x = self.fc1(x)
            x = self.fc2(x)
            v = self.out_v(x)
            return v


class PPOBuffer:
    def __init__(self, act_dim, obs_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, 3), dtype=np.float32)  # v, theta, task
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        # assert self.ptr < self.max_size  # buffer has to have room
        if self.ptr >= self.max_size:  # 缓冲区已满
            self.finish_path()  # 计算当前路径的 advantage 和 return
            self.ptr, self.path_start_idx = 0, 0  # 清空缓冲区
        self.obs_buf[self.ptr] = np.array(obs).reshape(-1)
        self.act_buf[self.ptr] = np.array(act).reshape(-1)
        self.rew_buf[self.ptr] = rew  # rew 是 float
        self.val_buf[self.ptr] = val  # val 是 float
        self.logp_buf[self.ptr] = logp  # logp 是 float

        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage estimation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)

        # Compute rewards-to-go as target value
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        # 不再强制 buffer 满
        assert self.ptr > 0  # 至少有一条数据
        path_slice = slice(0, self.ptr)

        # Normalize the advantages
        adv_mean = np.mean(self.adv_buf[path_slice])
        adv_std = np.std(self.adv_buf[path_slice]) + 1e-8
        self.adv_buf[path_slice] = (self.adv_buf[path_slice] - adv_mean) / adv_std

        with tf.device('/GPU:0'):
            data = [
                tf.convert_to_tensor(self.obs_buf[path_slice]),
                tf.convert_to_tensor(self.act_buf[path_slice]),
                tf.convert_to_tensor(self.adv_buf[path_slice]),
                tf.convert_to_tensor(self.ret_buf[path_slice]),
                tf.convert_to_tensor(self.logp_buf[path_slice])
            ]

        self.ptr, self.path_start_idx = 0, 0  # 清空 buffer
        return data

    @staticmethod
    def discount_cumsum(x, discount):
        return np.array([np.sum([discount ** t * x[t + i] for t in range(len(x) - i)]) for i in range(len(x))])


class MAPPO:
    def __init__(self, env):
        self.env = env
        self.n_agents = env.num_uav
        self.n_tasks = env.num_task
        # self.act_dim = 3  # [v, theta, task_logits]
        self.obs_dim = len(env._get_obs()[0])
        self.act_dim = 2 + TASK_NUM

        with tf.device('/GPU:0'):
            # Actor-Critic 网络
            self.actor = Actor(self.obs_dim)
            self.critic = Critic(self.obs_dim)

            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # 为每个agent设置一个buffer
        self.buffers = [PPOBuffer(obs_dim=self.obs_dim,
                                  act_dim=self.act_dim,
                                  size=4000,  # 每个agent最多步数
                                  gamma=0.99,
                                  lam=0.95) for _ in range(self.n_agents)]

        self.clip_ratio = 0.2
        self.train_iters = 80
        self.target_kl = 0.01

        self.uav_trajectories = [[] for _ in range(self.n_agents)]
        self.best_task_rate = 0  # 记录最好结果
        self.finish_time = 0  # 记录完成时间
        # 保存模型Checkpointing
        self.ckpt_dir = "MAPPO_checkpoints"
        self.actor_ckpt = os.path.join(self.ckpt_dir, "actor")
        self.critic_ckpt = os.path.join(self.ckpt_dir, "critic")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save_models(self):
        """保存模型到指定目录"""
        self.actor.save_weights(self.actor_ckpt)
        self.critic.save_weights(self.critic_ckpt)
        # save_model(self.actor, self.actor_ckpt)
        # save_model(self.critic, self.critic_ckpt)

        print("Models saved!")

    def load_models(self, epoch=None):
        """从指定目录加载模型"""
        self.actor.load_weights(self.actor_ckpt)
        self.critic.load_weights(self.critic_ckpt)
        print("Models loaded!")

    def get_action(self, obs):
        with tf.device('/GPU:0'):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)
            v_mean, theta_mean, task_logits = self.actor(obs)

            # Sample from normal distributions manually
            v_noise = tf.random.normal(shape=v_mean.shape, mean=0.0, stddev=0.1)
            theta_noise = tf.random.normal(shape=theta_mean.shape, mean=0.0, stddev=0.5)
            v_sample = tf.clip_by_value(v_mean + v_noise, 0.0, 1.0)
            v_scaled = v_sample * V_MAX  # 将速度改到0~V_MAX范围内
            theta_sample = tf.clip_by_value(theta_mean + theta_noise, -1.0, 1.0)
            theta_scaled = theta_sample * np.pi  # 将速度改到-pi到pi范围内
            task_sample = tf.random.categorical(task_logits, num_samples=1)

            # Compute log probabilities
            logp_v = -((v_sample - v_mean) ** 2) / (2 * 0.1 ** 2)
            logp_theta = -((theta_sample - theta_mean) ** 2) / (2 * 0.5 ** 2)
            logp_task = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=task_logits,
                                                                        labels=tf.squeeze(task_sample, axis=1))
            logp = tf.squeeze(logp_v) + tf.squeeze(logp_theta) + logp_task

            task_sample_float = tf.cast(task_sample, tf.float32) # tf.concat() 要求所有输入张量的数据类型必须一致，而 v_scaled 和 theta_scaled 是 float32 类型，但 task_sample_int 是 int32 类型
            act = tf.concat([v_scaled, theta_scaled, task_sample_float], axis=1)
            return act.numpy(), logp.numpy(), self.critic(obs).numpy()

    def compute_loss_pi(self, data):
        with tf.device('/GPU:0'):
            obs, act, adv, _, logp_old = data

            v_mean, theta_mean, task_logits = self.actor(obs)
            v_act = tf.expand_dims(act[:, 0], -1) / V_MAX  # 缩回 [0,1] 范围用于计算概率
            theta_act = tf.expand_dims(act[:, 1], -1) / np.pi
            task_act = tf.cast(act[:, 2], tf.int32)

            logp_v = -((v_act - v_mean) ** 2) / (2 * 0.1 ** 2)
            logp_theta = -((theta_act - theta_mean) ** 2) / (2 * 0.5 ** 2)
            logp_task = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=task_logits, labels=task_act)

            logp = tf.squeeze(logp_v) + tf.squeeze(logp_theta) + logp_task

            ratio = tf.exp(logp - logp_old)
            clip_adv = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_pi = -tf.reduce_mean(tf.minimum(ratio * adv, clip_adv))
            return loss_pi

    def compute_loss_v(self, data):
        with tf.device('/GPU:0'):
            obs, _, _, ret, _ = data  # 只用 obs 和 ret
            return tf.reduce_mean((self.critic(obs) - ret) ** 2)

    def update_with_data(self, data):
        with tf.device('/GPU:0'):
            for _ in range(self.train_iters):
                with tf.GradientTape() as tape:
                    loss_pi = self.compute_loss_pi(data)
                grads = tape.gradient(loss_pi, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

                with tf.GradientTape() as tape:
                    loss_v = self.compute_loss_v(data)
                grads = tape.gradient(loss_v, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    def train(self, epochs=100):
        all_reward = []
        all_completion = []
        all_collision = []
        all_fairness = []
        for ep in range(epochs):
            # 记录开始时间
            start_time = time.time()

            obs = self.env.reset(self.env.time_window_type)
            ep_len = 0
            episode_reward = np.zeros(UAV_NUM)
            done = False

            while not done:
                act, logp, val = self.get_action(obs)
                next_obs, rew, done = self.env.step(act)
                for i in range(self.n_agents):
                    self.buffers[i].store(obs[i], act[i], rew[i], val[i], logp[i])
                obs = next_obs
                episode_reward += rew
                ep_len += 1

            # 记录本轮的统计数据
            all_reward.append(np.sum(episode_reward))
            tasks_completed = self.env.tasks_completed / TASK_NUM
            all_completion.append(tasks_completed)
            collision = self.env.count_collisions
            all_collision.append(collision)
            fairness = self.env.calculate_fairness()
            all_fairness.append(fairness)

            if self.best_task_rate <= tasks_completed:
                self.best_task_rate = tasks_completed
                self.uav_trajectories = self.env.uav_trajectories
                self.finish_time = self.env.finish_time
            for i in range(self.n_agents):
                self.buffers[i].finish_path(last_val=0)
                data = self.buffers[i].get()
                self.update_with_data(data)

            print(f"回合 {ep}, 平均奖励: {np.mean(episode_reward):.2f}, "
                  f"任务完成率: {tasks_completed:.2f}, 碰撞次数: {collision}, 能耗均衡: {fairness}")

            # 记录本次运行的所有回合数据
            # 记录结束时间
            end_time = time.time()
            # 计算间隔时间
            interval = end_time - start_time
            # 打印间隔时间
            print(f"运行 {ep} 完成，耗时 {interval:.2f} 秒")
        self.save_models()
        return all_reward, all_completion, all_collision, all_fairness

    def test(self, episodes=10):
        self.load_models()
        all_reward = []
        all_completion = []
        all_collision = []
        all_energy = []
        for ep in range(episodes):
            # 记录开始时间
            start_time = time.time()

            obs = self.env.reset(self.env.time_window_type)
            ep_ret = 0
            done = np.zeros(self.n_agents)

            while not np.all(done):
                act, logp, val = self.get_action(obs)
                obs, rew, done, info = self.env.step(act)
                ep_ret = np.mean(rew)

            # 记录额外指标
            all_reward.append(ep_ret)
            all_completion.append(info['tasks_completed'])
            all_collision.append(info['collision'])
            all_energy.append(np.sum(info['energy']))

            print(
                f"Epoch {ep + 1:03d} | Return: {ep_ret:.3f} | Tasks Completed: {info['tasks_completed']:.2f}"
                f" | Collisions: {info['collision']}")

            # 记录本次运行的所有回合数据
            # 记录结束时间
            end_time = time.time()
            # 计算间隔时间
            interval = end_time - start_time
            # 打印间隔时间
            print(f"运行 {ep} 完成，耗时 {interval:.2f} 秒")
        return all_reward, all_completion, all_collision, all_energy
