# MADDPFG
UAV_NUM = 1  # 5
TASK_NUM = 5  # 10
TIME_STEPS = 100  # 150#100
MAP_SIZE = 100  # 500
E_MAX = 500
V_MAX = 10  # 15#35#20#30  # UAV最大移动速度 30m/s
D_SAFE = 5#15  # 10.0
XI = 0.01  # 0.04  # 感知衰减参数 ξ
D_MAX = 6  # 6#10.0#13.0#15.0#30.0  # 最大感知距离 d_max
P_TH = 0.8#0.6  # 0.8      # 群体成功概率阈值
LAMBDA = 0.05
G = 10.0#1.0
W = 5.0

A1 = 10#50#150  # 完成感知
A2 = 5#20#50  # 靠近任务点
A3 = 5.0#7.0  # 能耗
A4 = 5.0#14.0#7.0#30.0  # 碰撞
A5 = 1.0#5.0#15.0  # 协同
A6 = 0.0  # R_attract
A7 = 1.0#10.0  #边界
A8 = 1.0#8.0#5.0  # R_Fairness
A9 = 0  # 方向驱动

D_T = 1.0
H = 100

TX_POWER = 10  # 传输功率 (单位: W)
ANTENNA_GAIN = 1  # 天线增益 (单位: 线性倍数)
NOISE_POWER = 1e-10  # 噪声功率谱密度 (单位: W/Hz)
BANDWIDTH = 1e6  # 信道带宽 (单位: Hz)
PATH_LOSS_ALPHA = 2.0  # 路径损耗因子

# GA
# UAV_NUM = 3#5
# TASK_NUM = 15#10
# TIME_STEPS = 100#50
# MAP_SIZE = 100#500
# P_REQ = 1.0#1.0
# E_MAX = 500
# V_MAX = 10 #UAV最大移动速度 30m/s
# D_SAFE = 10#2.0
# XI = 0.04       # 感知衰减参数 ξ
# D_MAX = 30.0    # 最大感知距离 d_max
# P_TH = 0.6#0.8      # 群体成功概率阈值
# LAMBDA = 0.05
# G = 1.0
# W = 5.0
# A1 = 120.0
# A2 = 150.0#150.0
# A3 = 20
# A4 = 0.2#1.0
# A5 = 15.0
# A6 = 30.0
# A7 = 10.0#3.0
# A8 = 20.0
# D_T = 1.0
# H=100
#
# TX_POWER = 10          # 传输功率 (单位: W)
# ANTENNA_GAIN = 1       # 天线增益 (单位: 线性倍数)
# NOISE_POWER = 1e-10    # 噪声功率谱密度 (单位: W/Hz)
# BANDWIDTH = 1e6        # 信道带宽 (单位: Hz)
# PATH_LOSS_ALPHA = 2.0  # 路径损耗因子

# MAPPO
# UAV_NUM = 3#5
# TASK_NUM = 15#10
# TIME_STEPS = 100#50
# MAP_SIZE = 100#500
# P_REQ = 1.0#1.0
# E_MAX = 500
# V_MAX = 30 #UAV最大移动速度 30m/s
# D_SAFE = 10#2.0
# XI = 0.04       # 感知衰减参数 ξ
# D_MAX = 30.0    # 最大感知距离 d_max
# P_TH = 0.6#0.8      # 群体成功概率阈值
# LAMBDA = 0.05
# G = 1.0
# W = 5.0
#
# A1 = 200#150#120.0
# A2 = 200.0#150.0
# A3 = 0#0.000001
# A4 = 0.5#0.2#1.0
# A5 = 15.0#20.0#15.0
# A6 = 50.0
# A7 = 12.0#10.0
# A8 = 20.0
#
# D_T = 1.0
# H=100
#
# TX_POWER = 10          # 传输功率 (单位: W)
# ANTENNA_GAIN = 1       # 天线增益 (单位: 线性倍数)
# NOISE_POWER = 1e-10    # 噪声功率谱密度 (单位: W/Hz)
# BANDWIDTH = 1e6        # 信道带宽 (单位: Hz)
# PATH_LOSS_ALPHA = 2.0  # 路径损耗因子
