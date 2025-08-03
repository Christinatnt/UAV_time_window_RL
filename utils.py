import numpy as np
from config import *
from config import *


def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def jain_index(values):
    s = np.sum(values)
    sq = np.sum(np.square(values))
    return s ** 2 / (len(values) * sq + 1e-6)


def remain_time(t, task_window):
    current_time = t  # 假设环境有当前时间记录
    start_t, end_t = task_window
    # 剩余时间比例（越接近截止时间值越小）
    remaining_ratio = max(0, (end_t - current_time) / TIME_STEPS) #(end_t - start_t))
    # remaining_ratio = max(0, (end_t - current_time) / (end_t - start_t))
    return remaining_ratio
