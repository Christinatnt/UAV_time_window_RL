import numpy as np
import tensorflow as tf
import random
from config import *
from collections import deque
from tensorflow.keras import backend as K

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
        self.out_speed = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3))
        self.out_angle = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3))
        self.out_task = tf.keras.layers.Dense(TASK_NUM, activation='softmax')

    def call(self, inputs):
        x = self.swish(self.ln1(self.fc1(inputs)))
        x = self.swish(self.ln2(self.fc2(x)))
        speed = self.out_speed(x)
        angle = self.out_angle(x)
        task = self.out_task(x)
        return tf.concat([speed, angle, task], axis=-1)


class Critic(tf.keras.Model):
    def __init__(self, obs_all_dim, act_all_dim):
        super(Critic, self).__init__()
        self.obs_fc = tf.keras.layers.Dense(400, kernel_initializer='he_uniform')
        self.obs_ln = tf.keras.layers.LayerNormalization()
        self.concat_ln = tf.keras.layers.LayerNormalization()
        self.concat_fc = tf.keras.layers.Dense(300, kernel_initializer='he_uniform')
        self.out_fc = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomUniform(-3e-3, 3e-3))
        self.total_input_dim = 400 + act_all_dim  # 明确期望输入维度

    def call(self, obs, act):
        obs = tf.cast(obs, tf.float32)
        act = tf.cast(act, tf.float32)
        x_obs = tf.nn.relu(self.obs_ln(self.obs_fc(obs)))
        x = tf.concat([x_obs, act], axis=-1)
        assert x.shape[-1] == self.total_input_dim, f"[Critic] Input dim mismatch: got {x.shape[-1]}, expected {self.total_input_dim}"
        x = tf.nn.relu(self.concat_ln(self.concat_fc(x)))
        return self.out_fc(x)




class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        obs, acts, rews, next_obs, dones = map(np.array, zip(*batch))
        # 全部转换为 float32
        return (obs.astype(np.float32),
                acts.astype(np.float32),
                rews.astype(np.float32),
                next_obs.astype(np.float32),
                dones.astype(np.float32))

    def __len__(self):
        return len(self.buffer)


class MADDPG:
    def __init__(self, env):
        self.num_agents = env.num_uav
        self.obs_dim = len(env._get_obs()[0])
        self.act_dim = 2 + TASK_NUM  # 不是 3，而是 [speed, angle, task_prob]
        self.train_step = 0
        self.actors = [Actor(self.obs_dim, self.act_dim) for _ in range(self.num_agents)]


        self.actor_optim = [tf.keras.optimizers.Adam(1e-4) for _ in range(self.num_agents)]#原来为1e-3
        self.critic_optim = [tf.keras.optimizers.Adam(1e-3) for _ in range(self.num_agents)]#原来为1e-3

        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.95
        self.batch_size = 256#256#64

        # 初始化目标网络
        self.target_actors = [Actor(self.obs_dim, self.act_dim) for _ in range(self.num_agents)]
        obs_all_dim = self.obs_dim * self.num_agents
        act_all_dim = self.act_dim * self.num_agents

        self.critics = [Critic(obs_all_dim, act_all_dim) for _ in range(self.num_agents)]
        self.target_critics = [Critic(obs_all_dim, act_all_dim) for _ in range(self.num_agents)]

        self.noise_std = 0.1#0.2  # 控制随机性幅度
        self.epsilon = 1.0  # 初始探索率为 1.0，逐步衰减
        self.epsilon_min = 0.1#0.05
        self.epsilon_decay = 0.997#0.995

        # 拷贝权重
        for i in range(self.num_agents):
            self.update_target_network(self.target_actors[i], self.actors[i])
            self.update_target_network(self.target_critics[i], self.critics[i])


    '''def act(self, obs):
        actions_for_env = []
        actions_for_training = []
        for i, actor in enumerate(self.actors):
            obs_i = tf.convert_to_tensor([obs[i]], dtype=tf.float32)
            output = actor(obs_i).numpy()[0]  # shape = (2 + TASK_NUM, )

            # 加噪探索
            if np.random.rand() < self.epsilon:
                speed = np.random.uniform(0, 1) * V_MAX
                angle = np.random.uniform(-1, 1) * np.pi
                task_id = np.random.choice(TASK_NUM)
            else:
                speed = np.clip(output[0] + np.random.normal(0, self.noise_std), 0, 1) * V_MAX
                angle = np.clip(output[1] + np.random.normal(0, self.noise_std), -1, 1) * np.pi
                task_probs = np.clip(output[2:], 1e-8, 1.0)
                # 屏蔽已完成任务（假设你将 env 注入 maddpg 实例为 self.env）
                unfinished_mask = np.array([1.0 if self.env.task_done[k] < P_REQ else 0.0 for k in range(TASK_NUM)])
                task_probs *= unfinished_mask
                if task_probs.sum() > 0:
                    task_probs /= task_probs.sum()
                else:
                    task_probs = np.ones(TASK_NUM) / TASK_NUM  # 所有都完成，随机选
                task_id = np.random.choice(TASK_NUM, p=task_probs)

            actions_for_env.append([speed, angle, task_id])
            actions_for_training.append(output)  # 保存完整动作输出

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return actions_for_env, actions_for_training'''


    def act(self, obs):
        actions_for_env = []
        actions_for_train = []
        for i, actor in enumerate(self.actors):
            obs_i = tf.convert_to_tensor([obs[i]], dtype=tf.float32)
            output = actor(obs_i).numpy()[0]  # (2 + TASK_NUM, )

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

                unfinished_mask = np.array([1.0 if self.env.task_done[k] < P_REQ else 0.0 for k in range(TASK_NUM)])
                task_probs *= unfinished_mask
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

    def learn(self, obs, actions, rewards, next_obs, dones):
        # Sample a batch from replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough data

        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        # 确保维度合法
        assert obs_batch.shape == (self.batch_size, self.num_agents, self.obs_dim)
        assert act_batch.shape == (self.batch_size, self.num_agents, self.act_dim)

        for i in range(self.num_agents):
            # Prepare data for agent i
            obs_i = tf.convert_to_tensor(obs_batch[:, i], dtype=tf.float32)
            act_i = tf.convert_to_tensor(act_batch[:, i], dtype=tf.float32)
            rew_i = tf.convert_to_tensor([r[i] for r in rew_batch], dtype=tf.float32)
            next_obs_i = tf.convert_to_tensor(next_obs_batch[:, i], dtype=tf.float32)

            # === Critic Update ===
            with tf.GradientTape() as tape:
                # Get target actions for all agents
                target_acts = []
                for j in range(self.num_agents):
                    target_act_j = self.target_actors[j](tf.convert_to_tensor(next_obs_batch[:, j], dtype=tf.float32))
                    target_acts.append(target_act_j)
                target_acts_concat = tf.concat(target_acts, axis=-1)  # 这里必须保证每个 target_act_j 是 [256, 3]
                next_obs_concat = tf.reshape(next_obs_batch, [self.batch_size, -1])

                target_q = self.target_critics[i](next_obs_concat, target_acts_concat)
                y = rew_i + self.gamma * (1.0 - done_batch[:, i]) * tf.squeeze(target_q)
                y = tf.clip_by_value(y, -1000.0, 100.0)#TODO

                all_obs = tf.reshape(obs_batch, [self.batch_size, -1])
                # 正确动作维度是 num_agents * act_dim = 5 * 3 = 15
                all_acts = tf.reshape(act_batch, [self.batch_size, self.num_agents * self.act_dim])

                q_values = self.critics[i](all_obs, all_acts)
                q_values = tf.squeeze(q_values)

                critic_loss = tf.keras.losses.Huber()(y, q_values)

            critic_grads = tape.gradient(critic_loss, self.critics[i].trainable_variables)
            self.critic_optim[i].apply_gradients(zip(critic_grads, self.critics[i].trainable_variables))

            # === Actor Update ===
            with tf.GradientTape() as tape:
                # Generate new actions from current actor
                new_actions = []
                for j in range(self.num_agents):
                    obs_j = tf.convert_to_tensor(obs_batch[:, j], dtype=tf.float32)
                    act_j = self.actors[j](obs_j)
                    new_actions.append(act_j)
                all_new_actions = tf.concat(new_actions, axis=-1)

                all_obs = tf.reshape(obs_batch, [self.batch_size, -1])
                actor_loss = -tf.reduce_mean(self.critics[i](all_obs, all_new_actions))

            actor_grads = tape.gradient(actor_loss, self.actors[i].trainable_variables)
            self.actor_optim[i].apply_gradients(zip(actor_grads, self.actors[i].trainable_variables))

            # === Update target networks ===
            # 只每隔几步进行软更新（推荐）
            if self.train_step % 10 == 0:
                for i in range(self.num_agents):
                    self.update_target_network(self.target_actors[i], self.actors[i], tau=0.01)#tau=0.01
                    self.update_target_network(self.target_critics[i], self.critics[i], tau=0.01)
            self.train_step += 1

        #print(f"[DEBUG] obs shape: {all_obs.shape}, act shape: {all_acts.shape}")

    def update_target_network(self, target, source, tau=0.01):  # tau=1.0 为硬拷贝，可改软更新;0.01为软更新
        for target_param, param in zip(target.trainable_variables, source.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)
