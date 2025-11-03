"""
配置文件 - 遗传算法和机器人参数配置

这个文件定义了所有重要的参数，包括：
1. 遗传算法参数（种群大小、进化代数等）
2. 机器人物理参数
3. 适应度评估权重（决定什么行为是"好"的）
"""

# ==================== 遗传算法参数 ====================
GA_CONFIG = {
    # 种群参数
    'population_size': 30,          # 种群大小 - 每代有30个不同的"大脑"
    'generations': 50,              # 进化代数 - 总共进化50代
    'elite_size': 3,                # 精英个体数量 - 保留最好的3个

    # 遗传操作参数
    'mutation_rate': 0.2,           # 变异率 - 20%的基因会变异
    'crossover_rate': 0.8,          # 交叉率 - 80%的概率进行基因交叉
    'mutation_strength': 0.3,       # 变异强度 - 变异时改变的幅度

    # 神经网络参数（这些权重就是"基因"）
    'input_size': 11,               # 输入：8个距离传感器 + 3个地面传感器
    'hidden_size': 8,               # 隐藏层神经元数量
    'output_size': 2,               # 输出：左轮速度 + 右轮速度

    # 适应度评估参数
    'simulation_time': 30.0,        # 每个机器人测试30秒
    'time_step': 32,                # Webots时间步长(ms)
}

# ==================== 机器人参数 ====================
ROBOT_CONFIG = {
    # 电机参数
    'max_speed': 6.28,              # 最大轮速 (rad/s)
    'wheel_radius': 0.0205,         # 轮子半径 (m)
    'axle_length': 0.052,           # 轴距 (m)

    # 传感器参数
    'num_distance_sensors': 8,      # 距离传感器数量
    'num_ground_sensors': 3,        # 地面传感器数量
    'sensor_max_value': 4096,       # 传感器最大值

    # 传感器名称
    'distance_sensor_names': [
        'ps0', 'ps1', 'ps2', 'ps3',
        'ps4', 'ps5', 'ps6', 'ps7'
    ],
    'ground_sensor_names': [
        'gs0', 'gs1', 'gs2'
    ],
}

# ==================== 适应度权重（核心！）====================
# 这些权重决定了什么样的行为是"好"的
# 遗传算法会根据这些权重来评估每个机器人的表现
FITNESS_WEIGHTS = {
    'distance_weight': 2.0,         # 行驶距离越远越好
    'speed_weight': 1.0,            # 速度越快越好
    'line_following_weight': 3.0,   # 跟随黑线越准确越好（最重要）
    'obstacle_avoidance_weight': 2.0,  # 避障能力权重
    'collision_penalty': -10.0,     # 碰到障碍物扣分（严重惩罚）
    'deviation_penalty': -1.0,      # 偏离轨道扣分
    'smooth_weight': 0.5,           # 运动越平滑越好（不要左右摇摆）
    'completion_bonus': 50.0,       # 完成一圈的奖励
}

# ==================== 文件路径 ====================
PATHS = {
    'model_dir': 'models',
    'best_model': 'models/best_model.pkl',
    'checkpoint': 'models/checkpoint_{}.pkl',
    'log_dir': 'logs',
    'log_file': 'logs/training_log.txt',
}

# ==================== 测试参数 ====================
TEST_CONFIG = {
    'num_laps': 1,                  # 测试圈数
    'max_test_time': 120.0,         # 最大测试时间(秒)
    'lap_detection_threshold': 0.9, # 完成一圈的检测阈值
}
"""
配置文件 - 遗传算法和机器人参数配置

这个文件定义了所有重要的参数，包括：
1. 遗传算法参数（种群大小、进化代数等）
2. 机器人物理参数
3. 适应度评估权重（决定什么行为是"好"的）
"""

# ==================== 遗传算法参数 ====================
GA_CONFIG = {
    # 种群参数
    'population_size': 30,          # 种群大小 - 每代有30个不同的"大脑"
    'generations': 50,              # 进化代数 - 总共进化50代
    'elite_size': 3,                # 精英个体数量 - 保留最好的3个
    
    # 遗传操作参数
    'mutation_rate': 0.2,           # 变异率 - 20%的基因会变异
    'crossover_rate': 0.8,          # 交叉率 - 80%的概率进行基因交叉
    'mutation_strength': 0.3,       # 变异强度 - 变异时改变的幅度
    
    # 神经网络参数（这些权重就是"基因"）
    'input_size': 11,               # 输入：8个距离传感器 + 3个地面传感器
    'hidden_size': 8,               # 隐藏层神经元数量
    'output_size': 2,               # 输出：左轮速度 + 右轮速度
    
    # 适应度评估参数
    'simulation_time': 30.0,        # 每个机器人测试30秒
    'time_step': 32,                # Webots时间步长(ms)
}

# ==================== 机器人参数 ====================
ROBOT_CONFIG = {
    # 电机参数
    'max_speed': 6.28,              # 最大轮速 (rad/s)
    'wheel_radius': 0.0205,         # 轮子半径 (m)
    'axle_length': 0.052,           # 轴距 (m)
    
    # 传感器参数
    'num_distance_sensors': 8,      # 距离传感器数量
    'num_ground_sensors': 3,        # 地面传感器数量
    'sensor_max_value': 4096,       # 传感器最大值
    
    # 传感器名称
    'distance_sensor_names': [
        'ps0', 'ps1', 'ps2', 'ps3',
        'ps4', 'ps5', 'ps6', 'ps7'
    ],
    'ground_sensor_names': [
        'gs0', 'gs1', 'gs2'
    ],
}

# ==================== 适应度权重（核心！）====================
# 这些权重决定了什么样的行为是"好"的
# 遗传算法会根据这些权重来评估每个机器人的表现
FITNESS_WEIGHTS = {
    'distance_weight': 2.0,         # 行驶距离越远越好
    'speed_weight': 1.0,            # 速度越快越好
    'line_following_weight': 3.0,   # 跟随黑线越准确越好（最重要）
    'obstacle_avoidance_weight': 2.0,  # 避障能力权重
    'collision_penalty': -10.0,     # 碰到障碍物扣分（严重惩罚）
    'deviation_penalty': -1.0,      # 偏离轨道扣分
    'smooth_weight': 0.5,           # 运动越平滑越好（不要左右摇摆）
    'completion_bonus': 50.0,       # 完成一圈的奖励
}

# ==================== 文件路径 ====================
PATHS = {
    'model_dir': 'models',
    'best_model': 'models/best_model.pkl',
    'checkpoint': 'models/checkpoint_{}.pkl',
    'log_dir': 'logs',
    'log_file': 'logs/training_log.txt',
}

# ==================== 测试参数 ====================
TEST_CONFIG = {
    'num_laps': 1,                  # 测试圈数
    'max_test_time': 120.0,         # 最大测试时间(秒)
    'lap_detection_threshold': 0.9, # 完成一圈的检测阈值
}
