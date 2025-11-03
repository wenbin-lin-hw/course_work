"""
前向运动适应度函数 - 鼓励机器人在环形赛道上向前移动

这个模块专门设计用于环形赛道的循迹任务，目标是：
1. 鼓励机器人向前移动（沿着赛道方向）
2. 惩罚后退、原地打转等无效行为
3. 奖励高速平滑的前进运动
4. 考虑循迹准确度

关键概念：
- 前向速度：机器人沿着期望方向的速度分量
- 旋转惩罚：过度旋转会降低适应度
- 速度一致性：左右轮速度应该相近（直线前进）
"""
import numpy as np
import math


class ForwardFitnessEvaluator:
    """
    前向运动适应度评估器

    专门用于环形赛道，鼓励机器人沿着赛道向前移动
    """

    def __init__(self, max_speed=6.28):
        """
        初始化

        Args:
            max_speed: 电机最大速度（rad/s）
        """
        self.max_speed = max_speed

        # 累积指标
        self.total_forward_distance = 0.0    # 前向移动距离
        self.total_backward_distance = 0.0   # 后退距离
        self.total_rotation = 0.0            # 总旋转量
        self.total_speed = 0.0               # 总速度
        self.forward_speed_sum = 0.0         # 前向速度和

        # 运动质量指标
        self.speed_consistency_score = 0.0   # 速度一致性得分
        self.smooth_motion_score = 0.0       # 平滑运动得分

        # 历史数据
        self.prev_position = None
        self.prev_orientation = None
        self.prev_left_speed = 0.0
        self.prev_right_speed = 0.0

        # 计数器
        self.step_count = 0

    def reset(self):
        """重置评估器"""
        self.total_forward_distance = 0.0
        self.total_backward_distance = 0.0
        self.total_rotation = 0.0
        self.total_speed = 0.0
        self.forward_speed_sum = 0.0

        self.speed_consistency_score = 0.0
        self.smooth_motion_score = 0.0

        self.prev_position = None
        self.prev_orientation = None
        self.prev_left_speed = 0.0
        self.prev_right_speed = 0.0

        self.step_count = 0

    def _calculate_forward_component(self, position, prev_position, orientation):
        """
        计算前向移动分量

        这是核心函数！判断机器人是否在"向前"移动

        原理：
        1. 计算机器人的移动向量（从上一位置到当前位置）
        2. 计算机器人的朝向向量（机器人面向的方向）
        3. 计算两个向量的点积（投影）
        4. 点积 > 0：向前移动
           点积 < 0：向后移动
           点积 = 0：侧向移动

        Args:
            position: 当前位置 [x, y, z]
            prev_position: 上一位置 [x, y, z]
            orientation: 当前朝向角度（弧度）

        Returns:
            tuple: (前向距离, 总距离)
        """
        # 计算移动向量
        movement_vector = np.array([
            position[0] - prev_position[0],
            position[1] - prev_position[1]
        ])

        # 计算移动距离
        total_distance = np.linalg.norm(movement_vector)

        if total_distance < 1e-6:
            return 0.0, 0.0

        # 归一化移动向量
        movement_direction = movement_vector / total_distance

        # 计算机器人的朝向向量
        # orientation是机器人的朝向角度（弧度）
        # 朝向向量 = [cos(θ), sin(θ)]
        orientation_vector = np.array([
            math.cos(orientation),
            math.sin(orientation)
        ])

        # 计算点积（投影）
        # 点积 = |movement| * |orientation| * cos(夹角)
        # 因为两个向量都是单位向量，点积就是cos(夹角)
        dot_product = np.dot(movement_direction, orientation_vector)

        # 前向分量 = 总距离 × cos(夹角)
        forward_distance = total_distance * dot_product

        return forward_distance, total_distance

    def _calculate_rotation_amount(self, current_left_speed, current_right_speed):
        """
        计算旋转量

        原理：
        - 左右轮速度差越大，旋转越快
        - 旋转量 = |左轮速度 - 右轮速度| / 最大速度

        Args:
            current_left_speed: 当前左轮速度 [-1, 1]
            current_right_speed: 当前右轮速度 [-1, 1]

        Returns:
            float: 旋转量 [0, 1]
        """
        # 速度差的绝对值
        speed_diff = abs(current_left_speed - current_right_speed)

        # 归一化到 [0, 1]
        # 0 = 完全直行（左右轮速度相同）
        # 1 = 最大旋转（一个轮子全速前进，另一个全速后退）
        rotation = speed_diff / 2.0  # 除以2是因为速度范围是[-1, 1]

        return rotation

    def _calculate_speed_consistency(self, left_speed, right_speed):
        """
        计算速度一致性

        目标：鼓励左右轮速度相近（直线前进）

        原理：
        - 速度一致性 = 1 - |左轮 - 右轮| / 2
        - 完全一致（直行）= 1.0
        - 完全相反（原地转）= 0.0

        Args:
            left_speed: 左轮速度 [-1, 1]
            right_speed: 右轮速度 [-1, 1]

        Returns:
            float: 一致性得分 [0, 1]
        """
        speed_diff = abs(left_speed - right_speed)
        consistency = 1.0 - (speed_diff / 2.0)

        return max(0.0, consistency)

    def _calculate_smooth_motion(self, left_speed, right_speed,
                                  prev_left_speed, prev_right_speed):
        """
        计算运动平滑度

        目标：鼓励平滑的速度变化，避免突然加速/减速

        原理：
        - 计算速度变化量
        - 变化越小，平滑度越高

        Args:
            left_speed: 当前左轮速度
            right_speed: 当前右轮速度
            prev_left_speed: 上一步左轮速度
            prev_right_speed: 上一步右轮速度

        Returns:
            float: 平滑度得分 [0, 1]
        """
        # 计算速度变化
        left_change = abs(left_speed - prev_left_speed)
        right_change = abs(right_speed - prev_right_speed)

        # 平均变化量
        avg_change = (left_change + right_change) / 2.0

        # 平滑度得分（变化越小越好）
        # 使用指数衰减函数
        smoothness = math.exp(-avg_change * 5.0)  # 5.0是敏感度参数

        return smoothness

    def update(self, position, orientation, left_speed, right_speed):
        """
        更新适应度评估

        这是主要的更新函数，每个时间步调用一次

        Args:
            position: 当前位置 [x, y, z]
            orientation: 当前朝向角度（弧度）
            left_speed: 左轮速度 [-1, 1]
            right_speed: 右轮速度 [-1, 1]
        """
        self.step_count += 1

        # 第一步，只记录位置
        if self.prev_position is None:
            self.prev_position = position
            self.prev_orientation = orientation
            self.prev_left_speed = left_speed
            self.prev_right_speed = right_speed
            return

        # ========== 1. 计算前向移动分量 ==========
        forward_dist, total_dist = self._calculate_forward_component(
            position, self.prev_position, orientation
        )

        if forward_dist > 0:
            # 向前移动 - 好！
            self.total_forward_distance += forward_dist
        else:
            # 向后移动 - 不好！
            self.total_backward_distance += abs(forward_dist)

        # ========== 2. 计算旋转量 ==========
        rotation = self._calculate_rotation_amount(left_speed, right_speed)
        self.total_rotation += rotation

        # ========== 3. 计算速度 ==========
        avg_speed = (abs(left_speed) + abs(right_speed)) / 2.0
        self.total_speed += avg_speed

        # 前向速度（只有向前移动时才计入）
        if forward_dist > 0:
            self.forward_speed_sum += avg_speed

        # ========== 4. 计算速度一致性 ==========
        consistency = self._calculate_speed_consistency(left_speed, right_speed)
        self.speed_consistency_score += consistency

        # ========== 5. 计算运动平滑度 ==========
        smoothness = self._calculate_smooth_motion(
            left_speed, right_speed,
            self.prev_left_speed, self.prev_right_speed
        )
        self.smooth_motion_score += smoothness

        # 更新历史数据
        self.prev_position = position
        self.prev_orientation = orientation
        self.prev_left_speed = left_speed
        self.prev_right_speed = right_speed

    def calculate_fitness(self):
        """
        计算最终适应度分数

        这是核心的适应度函数！

        适应度 = 前向距离奖励
                + 速度奖励
                + 一致性奖励
                + 平滑度奖励
                - 后退惩罚
                - 旋转惩罚

        Returns:
            float: 适应度分数
        """
        if self.step_count == 0:
            return 0.0

        # 归一化各项指标
        avg_forward_distance = self.total_forward_distance
        avg_backward_distance = self.total_backward_distance
        avg_rotation = self.total_rotation / self.step_count
        avg_speed = self.total_speed / self.step_count
        avg_forward_speed = self.forward_speed_sum / max(1, self.step_count)
        avg_consistency = self.speed_consistency_score / self.step_count
        avg_smoothness = self.smooth_motion_score / self.step_count

        # ========== 计算适应度分数 ==========
        fitness = 0.0

        # 1. 前向距离奖励（最重要！）
        # 权重：5.0 - 走得越远越好
        fitness += 5.0 * avg_forward_distance

        # 2. 前向速度奖励
        # 权重：2.0 - 鼓励快速前进
        fitness += 2.0 * avg_forward_speed

        # 3. 速度一致性奖励
        # 权重：1.0 - 鼓励直线前进
        fitness += 1.0 * avg_consistency

        # 4. 平滑运动奖励
        # 权重：0.5 - 鼓励平滑控制
        fitness += 0.5 * avg_smoothness

        # 5. 后退惩罚
        # 权重：-3.0 - 严重惩罚后退
        fitness -= 3.0 * avg_backward_distance

        # 6. 过度旋转惩罚
        # 权重：-1.0 - 惩罚原地打转
        fitness -= 1.0 * avg_rotation

        # 7. 低速惩罚（鼓励快速移动）
        # 如果平均速度太低，给予惩罚
        if avg_speed < 0.3:
            fitness -= 2.0 * (0.3 - avg_speed)

        return max(0.0, fitness)  # 确保适应度非负

    def get_stats(self):
        """
        获取详细统计信息

        Returns:
            dict: 包含各项指标的字典
        """
        if self.step_count == 0:
            return {
                'fitness': 0.0,
                'forward_distance': 0.0,
                'backward_distance': 0.0,
                'net_distance': 0.0,
                'avg_speed': 0.0,
                'avg_forward_speed': 0.0,
                'avg_consistency': 0.0,
                'avg_smoothness': 0.0,
                'avg_rotation': 0.0,
                'forward_ratio': 0.0
            }

        return {
            'fitness': self.calculate_fitness(),
            'forward_distance': self.total_forward_distance,
            'backward_distance': self.total_backward_distance,
            'net_distance': self.total_forward_distance - self.total_backward_distance,
            'avg_speed': self.total_speed / self.step_count,
            'avg_forward_speed': self.forward_speed_sum / self.step_count,
            'avg_consistency': self.speed_consistency_score / self.step_count,
            'avg_smoothness': self.smooth_motion_score / self.step_count,
            'avg_rotation': self.total_rotation / self.step_count,
            'forward_ratio': self.total_forward_distance / max(0.001,
                self.total_forward_distance + self.total_backward_distance)
        }


# ==================== 简化版本 ====================

def simple_forward_fitness(left_speed, right_speed, forward_distance, backward_distance):
    """
    简化的前向适应度函数

    这是一个最简单的版本，适合快速测试

    Args:
        left_speed: 左轮速度 [-1, 1]
        right_speed: 右轮速度 [-1, 1]
        forward_distance: 前向移动距离
        backward_distance: 后退距离

    Returns:
        float: 适应度分数
    """
    # 1. 前向距离奖励
    fitness = 5.0 * forward_distance

    # 2. 速度奖励（鼓励快速移动）
    avg_speed = (abs(left_speed) + abs(right_speed)) / 2.0
    fitness += 2.0 * avg_speed

    # 3. 直线奖励（左右轮速度相近）
    speed_diff = abs(left_speed - right_speed)
    straightness = 1.0 - (speed_diff / 2.0)
    fitness += 1.0 * straightness

    # 4. 后退惩罚
    fitness -= 3.0 * backward_distance

    return max(0.0, fitness)


# ==================== 使用示例 ====================

def example_basic_usage():
    """
    示例1：基本使用方法
    """
    print("示例1: 基本使用")
    print("-" * 60)

    # 创建评估器
    evaluator = ForwardFitnessEvaluator(max_speed=6.28)

    # 模拟机器人运动（直线前进）
    print("\n场景1: 直线前进")
    evaluator.reset()

    positions = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ]

    orientation = 0.0  # 朝向右方（0度）
    left_speed = 0.8
    right_speed = 0.8

    for pos in positions:
        evaluator.update(pos, orientation, left_speed, right_speed)

    stats = evaluator.get_stats()
    print(f"适应度: {stats['fitness']:.2f}")
    print(f"前向距离: {stats['forward_distance']:.3f}m")
    print(f"后退距离: {stats['backward_distance']:.3f}m")
    print(f"速度一致性: {stats['avg_consistency']:.3f}")

    # 场景2: 后退
    print("\n场景2: 后退运动")
    evaluator.reset()

    positions = [
        [0.0, 0.0, 0.0],
        [-0.1, 0.0, 0.0],  # 向后移动
        [-0.2, 0.0, 0.0],
        [-0.3, 0.0, 0.0],
    ]

    orientation = 0.0  # 朝向右方
    left_speed = -0.8  # 负速度 = 后退
    right_speed = -0.8

    for pos in positions:
        evaluator.update(pos, orientation, left_speed, right_speed)

    stats = evaluator.get_stats()
    print(f"适应度: {stats['fitness']:.2f}")
    print(f"前向距离: {stats['forward_distance']:.3f}m")
    print(f"后退距离: {stats['backward_distance']:.3f}m")


def example_circular_track():
    """
    示例2：环形赛道上的运动
    """
    print("\n示例2: 环形赛道运动")
    print("-" * 60)

    evaluator = ForwardFitnessEvaluator()

    # 模拟机器人沿着圆形轨道移动
    radius = 1.0
    num_steps = 20

    print("\n沿着圆形轨道前进:")

    for i in range(num_steps):
        # 计算位置（圆周运动）
        angle = (2 * math.pi * i) / num_steps
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        position = [x, y, 0.0]

        # 朝向（切线方向）
        orientation = angle + math.pi / 2

        # 速度（稍微向左转）
        left_speed = 0.7
        right_speed = 0.8

        evaluator.update(position, orientation, left_speed, right_speed)

    stats = evaluator.get_stats()
    print(f"适应度: {stats['fitness']:.2f}")
    print(f"前向距离: {stats['forward_distance']:.3f}m")
    print(f"净距离: {stats['net_distance']:.3f}m")
    print(f"前向比例: {stats['forward_ratio']:.1%}")


def example_comparison():
    """
    示例3：不同行为的适应度对比
    """
    print("\n示例3: 不同行为的适应度对比")
    print("-" * 60)

    behaviors = [
        {
            'name': '快速直行',
            'left_speed': 0.9,
            'right_speed': 0.9,
            'movement': lambda i: [i * 0.15, 0, 0],
            'orientation': 0.0
        },
        {
            'name': '慢速直行',
            'left_speed': 0.3,
            'right_speed': 0.3,
            'movement': lambda i: [i * 0.05, 0, 0],
            'orientation': 0.0
        },
        {
            'name': '原地打转',
            'left_speed': 0.8,
            'right_speed': -0.8,
            'movement': lambda i: [0, 0, 0],
            'orientation': lambda i: i * 0.3
        },
        {
            'name': '后退',
            'left_speed': -0.8,
            'right_speed': -0.8,
            'movement': lambda i: [-i * 0.1, 0, 0],
            'orientation': 0.0
        },
        {
            'name': '曲线前进',
            'left_speed': 0.6,
            'right_speed': 0.9,
            'movement': lambda i: [i * 0.1, i * 0.05, 0],
            'orientation': lambda i: i * 0.1
        }
    ]

    print("\n行为对比:")
    print(f"{'行为':<12} {'适应度':<10} {'前向距离':<12} {'一致性':<10}")
    print("-" * 60)

    for behavior in behaviors:
        evaluator = ForwardFitnessEvaluator()

        for i in range(10):
            pos = behavior['movement'](i)
            orient = behavior['orientation'](i) if callable(behavior['orientation']) else behavior['orientation']
            evaluator.update(pos, orient, behavior['left_speed'], behavior['right_speed'])

        stats = evaluator.get_stats()
        print(f"{behavior['name']:<12} {stats['fitness']:<10.2f} "
              f"{stats['forward_distance']:<12.3f} {stats['avg_consistency']:<10.3f}")


def example_integration_with_ga():
    """
    示例4：与遗传算法集成
    """
    print("\n示例4: 与遗传算法集成")
    print("-" * 60)

    print("\n在遗传算法中使用:")
    print("""
# 在robot_controller.py中:

from forward_fitness import ForwardFitnessEvaluator

class EPuckController:
    def __init__(self, neural_network=None):
        # ... 其他初始化 ...
        
        # 创建前向适应度评估器
        self.forward_fitness = ForwardFitnessEvaluator()
    
    def step(self):
        # 1. 读取传感器
        sensor_data = self.read_sensors()
        
        # 2. 神经网络决策
        outputs = self.neural_network.forward(sensor_data)
        left_speed = outputs[0]
        right_speed = outputs[1]
        
        # 3. 获取位置和朝向
        position = self.get_position()
        orientation = self.get_orientation()
        
        # 4. 更新前向适应度评估器
        self.forward_fitness.update(
            position, 
            orientation, 
            left_speed, 
            right_speed
        )
        
        # 5. 设置电机
        self.set_motor_speeds(left_speed, right_speed)
    
    def get_fitness(self):
        # 返回前向适应度
        return self.forward_fitness.calculate_fitness()
    """)


if __name__ == '__main__':
    print("=" * 60)
    print("前向运动适应度函数示例")
    print("=" * 60)

    example_basic_usage()
    example_circular_track()
    example_comparison()
    example_integration_with_ga()

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
"""
前向运动适应度函数 - 鼓励机器人在环形赛道上向前移动

这个模块专门设计用于环形赛道的循迹任务，目标是：
1. 鼓励机器人向前移动（沿着赛道方向）
2. 惩罚后退、原地打转等无效行为
3. 奖励高速平滑的前进运动
4. 考虑循迹准确度

关键概念：
- 前向速度：机器人沿着期望方向的速度分量
- 旋转惩罚：过度旋转会降低适应度
- 速度一致性：左右轮速度应该相近（直线前进）
"""
import numpy as np
import math


class ForwardFitnessEvaluator:
    """
    前向运动适应度评估器
    
    专门用于环形赛道，鼓励机器人沿着赛道向前移动
    """
    
    def __init__(self, max_speed=6.28):
        """
        初始化
        
        Args:
            max_speed: 电机最大速度（rad/s）
        """
        self.max_speed = max_speed
        
        # 累积指标
        self.total_forward_distance = 0.0    # 前向移动距离
        self.total_backward_distance = 0.0   # 后退距离
        self.total_rotation = 0.0            # 总旋转量
        self.total_speed = 0.0               # 总速度
        self.forward_speed_sum = 0.0         # 前向速度和
        
        # 运动质量指标
        self.speed_consistency_score = 0.0   # 速度一致性得分
        self.smooth_motion_score = 0.0       # 平滑运动得分
        
        # 历史数据
        self.prev_position = None
        self.prev_orientation = None
        self.prev_left_speed = 0.0
        self.prev_right_speed = 0.0
        
        # 计数器
        self.step_count = 0
    
    def reset(self):
        """重置评估器"""
        self.total_forward_distance = 0.0
        self.total_backward_distance = 0.0
        self.total_rotation = 0.0
        self.total_speed = 0.0
        self.forward_speed_sum = 0.0
        
        self.speed_consistency_score = 0.0
        self.smooth_motion_score = 0.0
        
        self.prev_position = None
        self.prev_orientation = None
        self.prev_left_speed = 0.0
        self.prev_right_speed = 0.0
        
        self.step_count = 0
    
    def _calculate_forward_component(self, position, prev_position, orientation):
        """
        计算前向移动分量
        
        这是核心函数！判断机器人是否在"向前"移动
        
        原理：
        1. 计算机器人的移动向量（从上一位置到当前位置）
        2. 计算机器人的朝向向量（机器人面向的方向）
        3. 计算两个向量的点积（投影）
        4. 点积 > 0：向前移动
           点积 < 0：向后移动
           点积 = 0：侧向移动
        
        Args:
            position: 当前位置 [x, y, z]
            prev_position: 上一位置 [x, y, z]
            orientation: 当前朝向角度（弧度）
            
        Returns:
            tuple: (前向距离, 总距离)
        """
        # 计算移动向量
        movement_vector = np.array([
            position[0] - prev_position[0],
            position[1] - prev_position[1]
        ])
        
        # 计算移动距离
        total_distance = np.linalg.norm(movement_vector)
        
        if total_distance < 1e-6:
            return 0.0, 0.0
        
        # 归一化移动向量
        movement_direction = movement_vector / total_distance
        
        # 计算机器人的朝向向量
        # orientation是机器人的朝向角度（弧度）
        # 朝向向量 = [cos(θ), sin(θ)]
        orientation_vector = np.array([
            math.cos(orientation),
            math.sin(orientation)
        ])
        
        # 计算点积（投影）
        # 点积 = |movement| * |orientation| * cos(夹角)
        # 因为两个向量都是单位向量，点积就是cos(夹角)
        dot_product = np.dot(movement_direction, orientation_vector)
        
        # 前向分量 = 总距离 × cos(夹角)
        forward_distance = total_distance * dot_product
        
        return forward_distance, total_distance
    
    def _calculate_rotation_amount(self, current_left_speed, current_right_speed):
        """
        计算旋转量
        
        原理：
        - 左右轮速度差越大，旋转越快
        - 旋转量 = |左轮速度 - 右轮速度| / 最大速度
        
        Args:
            current_left_speed: 当前左轮速度 [-1, 1]
            current_right_speed: 当前右轮速度 [-1, 1]
            
        Returns:
            float: 旋转量 [0, 1]
        """
        # 速度差的绝对值
        speed_diff = abs(current_left_speed - current_right_speed)
        
        # 归一化到 [0, 1]
        # 0 = 完全直行（左右轮速度相同）
        # 1 = 最大旋转（一个轮子全速前进，另一个全速后退）
        rotation = speed_diff / 2.0  # 除以2是因为速度范围是[-1, 1]
        
        return rotation
    
    def _calculate_speed_consistency(self, left_speed, right_speed):
        """
        计算速度一致性
        
        目标：鼓励左右轮速度相近（直线前进）
        
        原理：
        - 速度一致性 = 1 - |左轮 - 右轮| / 2
        - 完全一致（直行）= 1.0
        - 完全相反（原地转）= 0.0
        
        Args:
            left_speed: 左轮速度 [-1, 1]
            right_speed: 右轮速度 [-1, 1]
            
        Returns:
            float: 一致性得分 [0, 1]
        """
        speed_diff = abs(left_speed - right_speed)
        consistency = 1.0 - (speed_diff / 2.0)
        
        return max(0.0, consistency)
    
    def _calculate_smooth_motion(self, left_speed, right_speed, 
                                  prev_left_speed, prev_right_speed):
        """
        计算运动平滑度
        
        目标：鼓励平滑的速度变化，避免突然加速/减速
        
        原理：
        - 计算速度变化量
        - 变化越小，平滑度越高
        
        Args:
            left_speed: 当前左轮速度
            right_speed: 当前右轮速度
            prev_left_speed: 上一步左轮速度
            prev_right_speed: 上一步右轮速度
            
        Returns:
            float: 平滑度得分 [0, 1]
        """
        # 计算速度变化
        left_change = abs(left_speed - prev_left_speed)
        right_change = abs(right_speed - prev_right_speed)
        
        # 平均变化量
        avg_change = (left_change + right_change) / 2.0
        
        # 平滑度得分（变化越小越好）
        # 使用指数衰减函数
        smoothness = math.exp(-avg_change * 5.0)  # 5.0是敏感度参数
        
        return smoothness
    
    def update(self, position, orientation, left_speed, right_speed):
        """
        更新适应度评估
        
        这是主要的更新函数，每个时间步调用一次
        
        Args:
            position: 当前位置 [x, y, z]
            orientation: 当前朝向角度（弧度）
            left_speed: 左轮速度 [-1, 1]
            right_speed: 右轮速度 [-1, 1]
        """
        self.step_count += 1
        
        # 第一步，只记录位置
        if self.prev_position is None:
            self.prev_position = position
            self.prev_orientation = orientation
            self.prev_left_speed = left_speed
            self.prev_right_speed = right_speed
            return
        
        # ========== 1. 计算前向移动分量 ==========
        forward_dist, total_dist = self._calculate_forward_component(
            position, self.prev_position, orientation
        )
        
        if forward_dist > 0:
            # 向前移动 - 好！
            self.total_forward_distance += forward_dist
        else:
            # 向后移动 - 不好！
            self.total_backward_distance += abs(forward_dist)
        
        # ========== 2. 计算旋转量 ==========
        rotation = self._calculate_rotation_amount(left_speed, right_speed)
        self.total_rotation += rotation
        
        # ========== 3. 计算速度 ==========
        avg_speed = (abs(left_speed) + abs(right_speed)) / 2.0
        self.total_speed += avg_speed
        
        # 前向速度（只有向前移动时才计入）
        if forward_dist > 0:
            self.forward_speed_sum += avg_speed
        
        # ========== 4. 计算速度一致性 ==========
        consistency = self._calculate_speed_consistency(left_speed, right_speed)
        self.speed_consistency_score += consistency
        
        # ========== 5. 计算运动平滑度 ==========
        smoothness = self._calculate_smooth_motion(
            left_speed, right_speed,
            self.prev_left_speed, self.prev_right_speed
        )
        self.smooth_motion_score += smoothness
        
        # 更新历史数据
        self.prev_position = position
        self.prev_orientation = orientation
        self.prev_left_speed = left_speed
        self.prev_right_speed = right_speed
    
    def calculate_fitness(self):
        """
        计算最终适应度分数
        
        这是核心的适应度函数！
        
        适应度 = 前向距离奖励 
                + 速度奖励 
                + 一致性奖励 
                + 平滑度奖励
                - 后退惩罚 
                - 旋转惩罚
        
        Returns:
            float: 适应度分数
        """
        if self.step_count == 0:
            return 0.0
        
        # 归一化各项指标
        avg_forward_distance = self.total_forward_distance
        avg_backward_distance = self.total_backward_distance
        avg_rotation = self.total_rotation / self.step_count
        avg_speed = self.total_speed / self.step_count
        avg_forward_speed = self.forward_speed_sum / max(1, self.step_count)
        avg_consistency = self.speed_consistency_score / self.step_count
        avg_smoothness = self.smooth_motion_score / self.step_count
        
        # ========== 计算适应度分数 ==========
        fitness = 0.0
        
        # 1. 前向距离奖励（最重要！）
        # 权重：5.0 - 走得越远越好
        fitness += 5.0 * avg_forward_distance
        
        # 2. 前向速度奖励
        # 权重：2.0 - 鼓励快速前进
        fitness += 2.0 * avg_forward_speed
        
        # 3. 速度一致性奖励
        # 权重：1.0 - 鼓励直线前进
        fitness += 1.0 * avg_consistency
        
        # 4. 平滑运动奖励
        # 权重：0.5 - 鼓励平滑控制
        fitness += 0.5 * avg_smoothness
        
        # 5. 后退惩罚
        # 权重：-3.0 - 严重惩罚后退
        fitness -= 3.0 * avg_backward_distance
        
        # 6. 过度旋转惩罚
        # 权重：-1.0 - 惩罚原地打转
        fitness -= 1.0 * avg_rotation
        
        # 7. 低速惩罚（鼓励快速移动）
        # 如果平均速度太低，给予惩罚
        if avg_speed < 0.3:
            fitness -= 2.0 * (0.3 - avg_speed)
        
        return max(0.0, fitness)  # 确保适应度非负
    
    def get_stats(self):
        """
        获取详细统计信息
        
        Returns:
            dict: 包含各项指标的字典
        """
        if self.step_count == 0:
            return {
                'fitness': 0.0,
                'forward_distance': 0.0,
                'backward_distance': 0.0,
                'net_distance': 0.0,
                'avg_speed': 0.0,
                'avg_forward_speed': 0.0,
                'avg_consistency': 0.0,
                'avg_smoothness': 0.0,
                'avg_rotation': 0.0,
                'forward_ratio': 0.0
            }
        
        return {
            'fitness': self.calculate_fitness(),
            'forward_distance': self.total_forward_distance,
            'backward_distance': self.total_backward_distance,
            'net_distance': self.total_forward_distance - self.total_backward_distance,
            'avg_speed': self.total_speed / self.step_count,
            'avg_forward_speed': self.forward_speed_sum / self.step_count,
            'avg_consistency': self.speed_consistency_score / self.step_count,
            'avg_smoothness': self.smooth_motion_score / self.step_count,
            'avg_rotation': self.total_rotation / self.step_count,
            'forward_ratio': self.total_forward_distance / max(0.001, 
                self.total_forward_distance + self.total_backward_distance)
        }


# ==================== 简化版本 ====================

def simple_forward_fitness(left_speed, right_speed, forward_distance, backward_distance):
    """
    简化的前向适应度函数
    
    这是一个最简单的版本，适合快速测试
    
    Args:
        left_speed: 左轮速度 [-1, 1]
        right_speed: 右轮速度 [-1, 1]
        forward_distance: 前向移动距离
        backward_distance: 后退距离
        
    Returns:
        float: 适应度分数
    """
    # 1. 前向距离奖励
    fitness = 5.0 * forward_distance
    
    # 2. 速度奖励（鼓励快速移动）
    avg_speed = (abs(left_speed) + abs(right_speed)) / 2.0
    fitness += 2.0 * avg_speed
    
    # 3. 直线奖励（左右轮速度相近）
    speed_diff = abs(left_speed - right_speed)
    straightness = 1.0 - (speed_diff / 2.0)
    fitness += 1.0 * straightness
    
    # 4. 后退惩罚
    fitness -= 3.0 * backward_distance
    
    return max(0.0, fitness)


# ==================== 使用示例 ====================

def example_basic_usage():
    """
    示例1：基本使用方法
    """
    print("示例1: 基本使用")
    print("-" * 60)
    
    # 创建评估器
    evaluator = ForwardFitnessEvaluator(max_speed=6.28)
    
    # 模拟机器人运动（直线前进）
    print("\n场景1: 直线前进")
    evaluator.reset()
    
    positions = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ]
    
    orientation = 0.0  # 朝向右方（0度）
    left_speed = 0.8
    right_speed = 0.8
    
    for pos in positions:
        evaluator.update(pos, orientation, left_speed, right_speed)
    
    stats = evaluator.get_stats()
    print(f"适应度: {stats['fitness']:.2f}")
    print(f"前向距离: {stats['forward_distance']:.3f}m")
    print(f"后退距离: {stats['backward_distance']:.3f}m")
    print(f"速度一致性: {stats['avg_consistency']:.3f}")
    
    # 场景2: 后退
    print("\n场景2: 后退运动")
    evaluator.reset()
    
    positions = [
        [0.0, 0.0, 0.0],
        [-0.1, 0.0, 0.0],  # 向后移动
        [-0.2, 0.0, 0.0],
        [-0.3, 0.0, 0.0],
    ]
    
    orientation = 0.0  # 朝向右方
    left_speed = -0.8  # 负速度 = 后退
    right_speed = -0.8
    
    for pos in positions:
        evaluator.update(pos, orientation, left_speed, right_speed)
    
    stats = evaluator.get_stats()
    print(f"适应度: {stats['fitness']:.2f}")
    print(f"前向距离: {stats['forward_distance']:.3f}m")
    print(f"后退距离: {stats['backward_distance']:.3f}m")


def example_circular_track():
    """
    示例2：环形赛道上的运动
    """
    print("\n示例2: 环形赛道运动")
    print("-" * 60)
    
    evaluator = ForwardFitnessEvaluator()
    
    # 模拟机器人沿着圆形轨道移动
    radius = 1.0
    num_steps = 20
    
    print("\n沿着圆形轨道前进:")
    
    for i in range(num_steps):
        # 计算位置（圆周运动）
        angle = (2 * math.pi * i) / num_steps
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        position = [x, y, 0.0]
        
        # 朝向（切线方向）
        orientation = angle + math.pi / 2
        
        # 速度（稍微向左转）
        left_speed = 0.7
        right_speed = 0.8
        
        evaluator.update(position, orientation, left_speed, right_speed)
    
    stats = evaluator.get_stats()
    print(f"适应度: {stats['fitness']:.2f}")
    print(f"前向距离: {stats['forward_distance']:.3f}m")
    print(f"净距离: {stats['net_distance']:.3f}m")
    print(f"前向比例: {stats['forward_ratio']:.1%}")


def example_comparison():
    """
    示例3：不同行为的适应度对比
    """
    print("\n示例3: 不同行为的适应度对比")
    print("-" * 60)
    
    behaviors = [
        {
            'name': '快速直行',
            'left_speed': 0.9,
            'right_speed': 0.9,
            'movement': lambda i: [i * 0.15, 0, 0],
            'orientation': 0.0
        },
        {
            'name': '慢速直行',
            'left_speed': 0.3,
            'right_speed': 0.3,
            'movement': lambda i: [i * 0.05, 0, 0],
            'orientation': 0.0
        },
        {
            'name': '原地打转',
            'left_speed': 0.8,
            'right_speed': -0.8,
            'movement': lambda i: [0, 0, 0],
            'orientation': lambda i: i * 0.3
        },
        {
            'name': '后退',
            'left_speed': -0.8,
            'right_speed': -0.8,
            'movement': lambda i: [-i * 0.1, 0, 0],
            'orientation': 0.0
        },
        {
            'name': '曲线前进',
            'left_speed': 0.6,
            'right_speed': 0.9,
            'movement': lambda i: [i * 0.1, i * 0.05, 0],
            'orientation': lambda i: i * 0.1
        }
    ]
    
    print("\n行为对比:")
    print(f"{'行为':<12} {'适应度':<10} {'前向距离':<12} {'一致性':<10}")
    print("-" * 60)
    
    for behavior in behaviors:
        evaluator = ForwardFitnessEvaluator()
        
        for i in range(10):
            pos = behavior['movement'](i)
            orient = behavior['orientation'](i) if callable(behavior['orientation']) else behavior['orientation']
            evaluator.update(pos, orient, behavior['left_speed'], behavior['right_speed'])
        
        stats = evaluator.get_stats()
        print(f"{behavior['name']:<12} {stats['fitness']:<10.2f} "
              f"{stats['forward_distance']:<12.3f} {stats['avg_consistency']:<10.3f}")


def example_integration_with_ga():
    """
    示例4：与遗传算法集成
    """
    print("\n示例4: 与遗传算法集成")
    print("-" * 60)
    
    print("\n在遗传算法中使用:")
    print("""
# 在robot_controller.py中:

from forward_fitness import ForwardFitnessEvaluator

class EPuckController:
    def __init__(self, neural_network=None):
        # ... 其他初始化 ...
        
        # 创建前向适应度评估器
        self.forward_fitness = ForwardFitnessEvaluator()
    
    def step(self):
        # 1. 读取传感器
        sensor_data = self.read_sensors()
        
        # 2. 神经网络决策
        outputs = self.neural_network.forward(sensor_data)
        left_speed = outputs[0]
        right_speed = outputs[1]
        
        # 3. 获取位置和朝向
        position = self.get_position()
        orientation = self.get_orientation()
        
        # 4. 更新前向适应度评估器
        self.forward_fitness.update(
            position, 
            orientation, 
            left_speed, 
            right_speed
        )
        
        # 5. 设置电机
        self.set_motor_speeds(left_speed, right_speed)
    
    def get_fitness(self):
        # 返回前向适应度
        return self.forward_fitness.calculate_fitness()
    """)


if __name__ == '__main__':
    print("=" * 60)
    print("前向运动适应度函数示例")
    print("=" * 60)
    
    # example_basic_usage()
    example_circular_track()
    # example_comparison()
    # example_integration_with_ga()
    #
    # print("\n" + "=" * 60)
    # print("完成！")
    # print("=" * 60)
