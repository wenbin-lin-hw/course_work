"""
适应度评估类 - 评估机器人表现的好坏

这是遗传算法的核心！
适应度函数决定了什么样的行为是"好"的，什么是"坏"的

评估指标：
1. 行驶距离 - 走得越远越好
2. 循迹准确度 - 跟随黑线越准确越好
3. 避障能力 - 成功避开障碍物
4. 运动平滑度 - 不要左右摇摆
5. 碰撞惩罚 - 撞到障碍物要扣分
6. 完成奖励 - 完成一圈有额外奖励
"""
import numpy as np
from config import FITNESS_WEIGHTS


class FitnessEvaluator:
    """
    适应度评估器

    根据机器人的行为计算适应度分数
    分数越高，机器人表现越好
    """

    def __init__(self):
        """初始化评估器"""
        self.weights = FITNESS_WEIGHTS
        self.reset()

    def reset(self):
        """重置评估器状态"""
        # 累计指标
        self.total_distance = 0.0           # 总行驶距离
        self.total_speed = 0.0              # 总速度
        self.line_following_score = 0.0     # 循迹得分
        self.obstacle_avoidance_score = 0.0 # 避障得分
        self.collision_count = 0            # 碰撞次数
        self.deviation_count = 0            # 偏离次数
        self.smoothness_score = 0.0         # 平滑度得分
        self.completed_lap = False          # 是否完成一圈

        # 历史数据（用于计算平滑度）
        self.prev_left_speed = 0.0
        self.prev_right_speed = 0.0
        self.prev_position = None

        # 计数器
        self.step_count = 0

    def update(self, sensor_data, motor_speeds, position):
        """
        更新评估指标

        每个时间步调用一次，累计机器人的表现数据

        Args:
            sensor_data: 传感器数据字典
                - 'distance_sensors': 8个距离传感器值
                - 'ground_sensors': 3个地面传感器值
            motor_speeds: [左轮速度, 右轮速度]
            position: [x, y, z] 机器人位置
        """
        self.step_count += 1

        # 1. 计算行驶距离
        if self.prev_position is not None:
            distance = np.linalg.norm(
                np.array(position[:2]) - np.array(self.prev_position[:2])
            )
            self.total_distance += distance
        self.prev_position = position

        # 2. 计算速度（鼓励快速前进）
        avg_speed = (abs(motor_speeds[0]) + abs(motor_speeds[1])) / 2.0
        self.total_speed += avg_speed

        # 3. 评估循迹准确度
        # 地面传感器：中间传感器应该检测到黑线
        ground_sensors = sensor_data['ground_sensors']
        # 归一化地面传感器值 [0, 1]，1表示检测到黑线
        gs_normalized = [gs / 1000.0 for gs in ground_sensors]

        # 如果中间传感器检测到黑线，得分高
        # 如果左右传感器检测到黑线，说明偏离了，得分低
        if len(gs_normalized) >= 3:
            center_score = gs_normalized[1]  # 中间传感器
            side_penalty = (gs_normalized[0] + gs_normalized[2]) / 2.0
            line_score = center_score - side_penalty * 0.5
            self.line_following_score += max(0, line_score)

        # 4. 评估避障能力
        # 距离传感器：前方传感器检测到障碍物时应该减速或转向
        distance_sensors = sensor_data['distance_sensors']
        # 归一化距离传感器值 [0, 1]，1表示检测到障碍物
        ds_normalized = [ds / 4096.0 for ds in distance_sensors]

        # 前方传感器（ps0, ps1, ps6, ps7）
        front_sensors = [ds_normalized[0], ds_normalized[1],
                        ds_normalized[6], ds_normalized[7]]
        front_obstacle = max(front_sensors)

        # 如果前方有障碍物且速度降低，说明在避障
        if front_obstacle > 0.3:  # 检测到障碍物
            if avg_speed < 0.5:  # 减速了
                self.obstacle_avoidance_score += 1.0
            else:  # 没减速，可能要碰撞
                self.collision_count += 1

        # 5. 检测碰撞（前方传感器值很高）
        if front_obstacle > 0.8:
            self.collision_count += 1

        # 6. 检测偏离轨道（所有地面传感器都检测不到黑线）
        if max(gs_normalized) < 0.2:
            self.deviation_count += 1

        # 7. 评估运动平滑度（速度变化不要太剧烈）
        speed_change = (
            abs(motor_speeds[0] - self.prev_left_speed) +
            abs(motor_speeds[1] - self.prev_right_speed)
        )
        smoothness = 1.0 / (1.0 + speed_change)  # 变化越小，平滑度越高
        self.smoothness_score += smoothness

        self.prev_left_speed = motor_speeds[0]
        self.prev_right_speed = motor_speeds[1]

    def check_lap_completion(self, position, start_position, threshold=0.5):
        """
        检查是否完成一圈

        Args:
            position: 当前位置 [x, y, z]
            start_position: 起始位置 [x, y, z]
            threshold: 距离阈值（米）

        Returns:
            bool: 是否完成一圈
        """
        if self.step_count < 100:  # 至少要运行一段时间
            return False

        distance_to_start = np.linalg.norm(
            np.array(position[:2]) - np.array(start_position[:2])
        )

        if distance_to_start < threshold and self.total_distance > 2.0:
            self.completed_lap = True
            return True

        return False

    def calculate_fitness(self):
        """
        计算最终适应度分数

        这是遗传算法的核心！
        根据各项指标和权重计算总分

        Returns:
            float: 适应度分数，越高越好
        """
        if self.step_count == 0:
            return 0.0

        # 归一化各项指标（除以步数得到平均值）
        avg_distance = self.total_distance
        avg_speed = self.total_speed / self.step_count
        avg_line_following = self.line_following_score / self.step_count
        avg_obstacle_avoidance = self.obstacle_avoidance_score / self.step_count
        avg_smoothness = self.smoothness_score / self.step_count

        # 计算加权得分
        fitness = 0.0

        # 正向奖励
        fitness += self.weights['distance_weight'] * avg_distance
        fitness += self.weights['speed_weight'] * avg_speed
        fitness += self.weights['line_following_weight'] * avg_line_following
        fitness += self.weights['obstacle_avoidance_weight'] * avg_obstacle_avoidance
        fitness += self.weights['smooth_weight'] * avg_smoothness

        # 负向惩罚
        fitness += self.weights['collision_penalty'] * self.collision_count
        fitness += self.weights['deviation_penalty'] * (self.deviation_count / self.step_count)

        # 完成奖励
        if self.completed_lap:
            fitness += self.weights['completion_bonus']

        return max(0.0, fitness)  # 确保适应度非负

    def get_stats(self):
        """
        获取详细统计信息

        Returns:
            dict: 包含各项指标的字典
        """
        return {
            'fitness': self.calculate_fitness(),
            'distance': self.total_distance,
            'avg_speed': self.total_speed / max(1, self.step_count),
            'line_following': self.line_following_score / max(1, self.step_count),
            'obstacle_avoidance': self.obstacle_avoidance_score / max(1, self.step_count),
            'collisions': self.collision_count,
            'deviations': self.deviation_count,
            'smoothness': self.smoothness_score / max(1, self.step_count),
            'completed_lap': self.completed_lap,
            'steps': self.step_count
        }
"""
适应度评估类 - 评估机器人表现的好坏

这是遗传算法的核心！
适应度函数决定了什么样的行为是"好"的，什么是"坏"的

评估指标：
1. 行驶距离 - 走得越远越好
2. 循迹准确度 - 跟随黑线越准确越好
3. 避障能力 - 成功避开障碍物
4. 运动平滑度 - 不要左右摇摆
5. 碰撞惩罚 - 撞到障碍物要扣分
6. 完成奖励 - 完成一圈有额外奖励
"""
import numpy as np
from config import FITNESS_WEIGHTS


class FitnessEvaluator:
    """
    适应度评估器
    
    根据机器人的行为计算适应度分数
    分数越高，机器人表现越好
    """
    
    def __init__(self):
        """初始化评估器"""
        self.weights = FITNESS_WEIGHTS
        self.reset()
    
    def reset(self):
        """重置评估器状态"""
        # 累计指标
        self.total_distance = 0.0           # 总行驶距离
        self.total_speed = 0.0              # 总速度
        self.line_following_score = 0.0     # 循迹得分
        self.obstacle_avoidance_score = 0.0 # 避障得分
        self.collision_count = 0            # 碰撞次数
        self.deviation_count = 0            # 偏离次数
        self.smoothness_score = 0.0         # 平滑度得分
        self.completed_lap = False          # 是否完成一圈
        
        # 历史数据（用于计算平滑度）
        self.prev_left_speed = 0.0
        self.prev_right_speed = 0.0
        self.prev_position = None
        
        # 计数器
        self.step_count = 0
    
    def update(self, sensor_data, motor_speeds, position):
        """
        更新评估指标
        
        每个时间步调用一次，累计机器人的表现数据
        
        Args:
            sensor_data: 传感器数据字典
                - 'distance_sensors': 8个距离传感器值
                - 'ground_sensors': 3个地面传感器值
            motor_speeds: [左轮速度, 右轮速度]
            position: [x, y, z] 机器人位置
        """
        self.step_count += 1
        
        # 1. 计算行驶距离
        if self.prev_position is not None:
            distance = np.linalg.norm(
                np.array(position[:2]) - np.array(self.prev_position[:2])
            )
            self.total_distance += distance
        self.prev_position = position
        
        # 2. 计算速度（鼓励快速前进）
        avg_speed = (abs(motor_speeds[0]) + abs(motor_speeds[1])) / 2.0
        self.total_speed += avg_speed
        
        # 3. 评估循迹准确度
        # 地面传感器：中间传感器应该检测到黑线
        ground_sensors = sensor_data['ground_sensors']
        # 归一化地面传感器值 [0, 1]，1表示检测到黑线
        gs_normalized = [gs / 1000.0 for gs in ground_sensors]
        
        # 如果中间传感器检测到黑线，得分高
        # 如果左右传感器检测到黑线，说明偏离了，得分低
        if len(gs_normalized) >= 3:
            center_score = gs_normalized[1]  # 中间传感器
            side_penalty = (gs_normalized[0] + gs_normalized[2]) / 2.0
            line_score = center_score - side_penalty * 0.5
            self.line_following_score += max(0, line_score)
        
        # 4. 评估避障能力
        # 距离传感器：前方传感器检测到障碍物时应该减速或转向
        distance_sensors = sensor_data['distance_sensors']
        # 归一化距离传感器值 [0, 1]，1表示检测到障碍物
        ds_normalized = [ds / 4096.0 for ds in distance_sensors]
        
        # 前方传感器（ps0, ps1, ps6, ps7）
        front_sensors = [ds_normalized[0], ds_normalized[1], 
                        ds_normalized[6], ds_normalized[7]]
        front_obstacle = max(front_sensors)
        
        # 如果前方有障碍物且速度降低，说明在避障
        if front_obstacle > 0.3:  # 检测到障碍物
            if avg_speed < 0.5:  # 减速了
                self.obstacle_avoidance_score += 1.0
            else:  # 没减速，可能要碰撞
                self.collision_count += 1
        
        # 5. 检测碰撞（前方传感器值很高）
        if front_obstacle > 0.8:
            self.collision_count += 1
        
        # 6. 检测偏离轨道（所有地面传感器都检测不到黑线）
        if max(gs_normalized) < 0.2:
            self.deviation_count += 1
        
        # 7. 评估运动平滑度（速度变化不要太剧烈）
        speed_change = (
            abs(motor_speeds[0] - self.prev_left_speed) +
            abs(motor_speeds[1] - self.prev_right_speed)
        )
        smoothness = 1.0 / (1.0 + speed_change)  # 变化越小，平滑度越高
        self.smoothness_score += smoothness
        
        self.prev_left_speed = motor_speeds[0]
        self.prev_right_speed = motor_speeds[1]
    
    def check_lap_completion(self, position, start_position, threshold=0.5):
        """
        检查是否完成一圈
        
        Args:
            position: 当前位置 [x, y, z]
            start_position: 起始位置 [x, y, z]
            threshold: 距离阈值（米）
            
        Returns:
            bool: 是否完成一圈
        """
        if self.step_count < 100:  # 至少要运行一段时间
            return False
        
        distance_to_start = np.linalg.norm(
            np.array(position[:2]) - np.array(start_position[:2])
        )
        
        if distance_to_start < threshold and self.total_distance > 2.0:
            self.completed_lap = True
            return True
        
        return False
    
    def calculate_fitness(self):
        """
        计算最终适应度分数
        
        这是遗传算法的核心！
        根据各项指标和权重计算总分
        
        Returns:
            float: 适应度分数，越高越好
        """
        if self.step_count == 0:
            return 0.0
        
        # 归一化各项指标（除以步数得到平均值）
        avg_distance = self.total_distance
        avg_speed = self.total_speed / self.step_count
        avg_line_following = self.line_following_score / self.step_count
        avg_obstacle_avoidance = self.obstacle_avoidance_score / self.step_count
        avg_smoothness = self.smoothness_score / self.step_count
        
        # 计算加权得分
        fitness = 0.0
        
        # 正向奖励
        fitness += self.weights['distance_weight'] * avg_distance
        fitness += self.weights['speed_weight'] * avg_speed
        fitness += self.weights['line_following_weight'] * avg_line_following
        fitness += self.weights['obstacle_avoidance_weight'] * avg_obstacle_avoidance
        fitness += self.weights['smooth_weight'] * avg_smoothness
        
        # 负向惩罚
        fitness += self.weights['collision_penalty'] * self.collision_count
        fitness += self.weights['deviation_penalty'] * (self.deviation_count / self.step_count)
        
        # 完成奖励
        if self.completed_lap:
            fitness += self.weights['completion_bonus']
        
        return max(0.0, fitness)  # 确保适应度非负
    
    def get_stats(self):
        """
        获取详细统计信息
        
        Returns:
            dict: 包含各项指标的字典
        """
        return {
            'fitness': self.calculate_fitness(),
            'distance': self.total_distance,
            'avg_speed': self.total_speed / max(1, self.step_count),
            'line_following': self.line_following_score / max(1, self.step_count),
            'obstacle_avoidance': self.obstacle_avoidance_score / max(1, self.step_count),
            'collisions': self.collision_count,
            'deviations': self.deviation_count,
            'smoothness': self.smoothness_score / max(1, self.step_count),
            'completed_lap': self.completed_lap,
            'steps': self.step_count
        }
