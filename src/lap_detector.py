"""
圈数检测和赛道长度计算模块

提供多种方法来：
1. 检测机器人是否完成一圈
2. 计算赛道的实际长度
3. 追踪机器人的圈数和进度

方法：
- 起点/终点检测法：检测机器人是否回到起点
- 扇区检测法：将赛道分成多个扇区，确保机器人按顺序通过
- 角度累积法：累积机器人转过的角度，360度=一圈
"""
import numpy as np
import math


class LapDetector:
    """
    圈数检测器基类
    """
    
    def __init__(self, start_position, threshold=0.3):
        """
        初始化
        
        Args:
            start_position: 起点位置 [x, y, z]
            threshold: 距离阈值（米），小于此距离认为回到起点
        """
        self.start_position = np.array(start_position[:2])  # 只用x,y
        self.threshold = threshold
        
        self.lap_count = 0
        self.lap_completed = False
        self.lap_distances = []  # 每圈的距离
        self.current_lap_distance = 0.0
        
        self.min_distance_before_finish = 2.0  # 完成一圈前必须走的最小距离
    
    def reset(self):
        """重置检测器"""
        self.lap_count = 0
        self.lap_completed = False
        self.lap_distances = []
        self.current_lap_distance = 0.0
    
    def update(self, current_position, distance_moved):
        """
        更新检测器
        
        Args:
            current_position: 当前位置 [x, y, z]
            distance_moved: 本次移动的距离
            
        Returns:
            bool: 是否完成了一圈
        """
        self.current_lap_distance += distance_moved
        
        # 检查是否回到起点
        distance_to_start = np.linalg.norm(
            np.array(current_position[:2]) - self.start_position
        )
        
        # 必须先走一定距离，才能检测完成一圈（避免误判）
        if (distance_to_start < self.threshold and 
            self.current_lap_distance > self.min_distance_before_finish):
            
            # 完成一圈！
            self.lap_count += 1
            self.lap_distances.append(self.current_lap_distance)
            self.lap_completed = True
            self.current_lap_distance = 0.0
            
            return True
        
        return False
    
    def get_lap_count(self):
        """获取完成的圈数"""
        return self.lap_count
    
    def get_average_lap_distance(self):
        """获取平均圈长"""
        if self.lap_distances:
            return np.mean(self.lap_distances)
        return 0.0
    
    def get_circuit_length(self):
        """获取赛道长度（基于已完成的圈数）"""
        return self.get_average_lap_distance()


class SectorLapDetector(LapDetector):
    """
    扇区检测器
    
    将赛道分成多个扇区，机器人必须按顺序通过所有扇区才算完成一圈
    这样可以防止机器人"作弊"（直接从起点附近绕回来）
    
    原理：
    - 将圆形赛道分成N个扇区（如4个：北、东、南、西）
    - 机器人必须依次通过所有扇区
    - 通过所有扇区后回到起点，才算完成一圈
    """
    
    def __init__(self, start_position, center_position=None, num_sectors=4, threshold=0.3):
        """
        初始化
        
        Args:
            start_position: 起点位置 [x, y, z]
            center_position: 赛道中心位置 [x, y, z]，如果为None则使用原点
            num_sectors: 扇区数量（建议4或8）
            threshold: 距离阈值
        """
        super().__init__(start_position, threshold)
        
        if center_position is None:
            center_position = [0, 0, 0]
        self.center = np.array(center_position[:2])
        
        self.num_sectors = num_sectors
        self.sectors_visited = set()  # 已访问的扇区
        self.current_sector = -1
        self.last_sector = -1
        
        # 计算起点所在的扇区
        self.start_sector = self._get_sector(start_position)
    
    def _get_sector(self, position):
        """
        计算位置所在的扇区
        
        Args:
            position: [x, y, z]
            
        Returns:
            int: 扇区编号 (0 到 num_sectors-1)
        """
        # 计算相对于中心的角度
        rel_pos = np.array(position[:2]) - self.center
        angle = math.atan2(rel_pos[1], rel_pos[0])  # 范围 [-π, π]
        
        # 转换到 [0, 2π]
        if angle < 0:
            angle += 2 * math.pi
        
        # 计算扇区编号
        sector = int(angle / (2 * math.pi / self.num_sectors))
        return sector % self.num_sectors
    
    def reset(self):
        """重置检测器"""
        super().reset()
        self.sectors_visited = set()
        self.current_sector = -1
        self.last_sector = -1
    
    def update(self, current_position, distance_moved):
        """
        更新检测器
        
        Args:
            current_position: 当前位置 [x, y, z]
            distance_moved: 本次移动的距离
            
        Returns:
            bool: 是否完成了一圈
        """
        self.current_lap_distance += distance_moved
        
        # 更新当前扇区
        self.current_sector = self._get_sector(current_position)
        
        # 记录访问过的扇区
        if self.current_sector != self.last_sector:
            self.sectors_visited.add(self.current_sector)
            self.last_sector = self.current_sector
        
        # 检查是否回到起点
        distance_to_start = np.linalg.norm(
            np.array(current_position[:2]) - self.start_position
        )
        
        # 完成一圈的条件：
        # 1. 回到起点附近
        # 2. 访问过所有扇区
        # 3. 走过最小距离
        if (distance_to_start < self.threshold and 
            len(self.sectors_visited) >= self.num_sectors and
            self.current_lap_distance > self.min_distance_before_finish):
            
            # 完成一圈！
            self.lap_count += 1
            self.lap_distances.append(self.current_lap_distance)
            self.lap_completed = True
            
            # 重置扇区记录
            self.sectors_visited = {self.start_sector}
            self.current_lap_distance = 0.0
            
            return True
        
        return False
    
    def get_progress(self):
        """
        获取当前圈的进度
        
        Returns:
            float: 进度百分比 (0.0 到 1.0)
        """
        return len(self.sectors_visited) / self.num_sectors


class AngleLapDetector(LapDetector):
    """
    角度累积检测器
    
    通过累积机器人转过的角度来检测圈数
    转过360度（2π弧度）= 完成一圈
    
    优点：不需要精确的起点位置
    缺点：需要准确的朝向信息
    """
    
    def __init__(self, start_position, threshold=0.3):
        """
        初始化
        
        Args:
            start_position: 起点位置 [x, y, z]
            threshold: 距离阈值
        """
        super().__init__(start_position, threshold)
        
        self.total_angle = 0.0  # 累积转过的角度
        self.prev_angle = None
    
    def reset(self):
        """重置检测器"""
        super().reset()
        self.total_angle = 0.0
        self.prev_angle = None
    
    def _get_angle(self, position, center=[0, 0]):
        """
        计算位置相对于中心的角度
        
        Args:
            position: [x, y, z]
            center: 中心位置 [x, y]
            
        Returns:
            float: 角度（弧度）
        """
        rel_pos = np.array(position[:2]) - np.array(center)
        angle = math.atan2(rel_pos[1], rel_pos[0])
        return angle
    
    def update(self, current_position, distance_moved):
        """
        更新检测器
        
        Args:
            current_position: 当前位置 [x, y, z]
            distance_moved: 本次移动的距离
            
        Returns:
            bool: 是否完成了一圈
        """
        self.current_lap_distance += distance_moved
        
        # 计算当前角度
        current_angle = self._get_angle(current_position)
        
        if self.prev_angle is not None:
            # 计算角度变化
            angle_delta = current_angle - self.prev_angle
            
            # 处理角度跳变（从π到-π或反之）
            if angle_delta > math.pi:
                angle_delta -= 2 * math.pi
            elif angle_delta < -math.pi:
                angle_delta += 2 * math.pi
            
            # 累积角度（取绝对值，因为可能逆时针或顺时针）
            self.total_angle += abs(angle_delta)
        
        self.prev_angle = current_angle
        
        # 检查是否转过360度
        if (self.total_angle >= 2 * math.pi and 
            self.current_lap_distance > self.min_distance_before_finish):
            
            # 完成一圈！
            self.lap_count += 1
            self.lap_distances.append(self.current_lap_distance)
            self.lap_completed = True
            
            # 重置角度
            self.total_angle = 0.0
            self.current_lap_distance = 0.0
            
            return True
        
        return False
    
    def get_progress(self):
        """
        获取当前圈的进度
        
        Returns:
            float: 进度百分比 (0.0 到 1.0)
        """
        return min(1.0, self.total_angle / (2 * math.pi))


class CircuitLengthEstimator:
    """
    赛道长度估算器
    
    通过多种方法估算赛道的实际长度
    """
    
    def __init__(self, center=[0, 0], radius=None):
        """
        初始化
        
        Args:
            center: 赛道中心 [x, y]
            radius: 赛道半径（如果已知）
        """
        self.center = np.array(center)
        self.radius = radius
        
        self.position_samples = []  # 位置样本
        self.distance_samples = []  # 距离样本
    
    def add_sample(self, position):
        """
        添加位置样本
        
        Args:
            position: [x, y, z]
        """
        self.position_samples.append(position[:2])
    
    def add_distance_sample(self, distance):
        """
        添加距离样本（完成一圈的距离）
        
        Args:
            distance: 圈长（米）
        """
        self.distance_samples.append(distance)
    
    def estimate_from_radius(self, radius=None):
        """
        从半径估算赛道长度
        
        Args:
            radius: 赛道半径，如果为None则使用初始化时的半径
            
        Returns:
            float: 估算的赛道长度
        """
        if radius is None:
            radius = self.radius
        
        if radius is None:
            raise ValueError("需要提供赛道半径")
        
        # 圆形赛道：周长 = 2πr
        return 2 * math.pi * radius
    
    def estimate_from_samples(self):
        """
        从位置样本估算赛道长度
        
        通过拟合圆来估算半径，然后计算周长
        
        Returns:
            float: 估算的赛道长度
        """
        if len(self.position_samples) < 3:
            raise ValueError("需要至少3个位置样本")
        
        # 计算样本点到中心的平均距离（估算半径）
        samples = np.array(self.position_samples)
        distances = np.linalg.norm(samples - self.center, axis=1)
        estimated_radius = np.mean(distances)
        
        # 计算周长
        return 2 * math.pi * estimated_radius
    
    def estimate_from_completed_laps(self):
        """
        从完成的圈数估算赛道长度
        
        使用实际测量的圈长的平均值
        
        Returns:
            float: 估算的赛道长度
        """
        if not self.distance_samples:
            raise ValueError("没有完成圈的距离样本")
        
        return np.mean(self.distance_samples)
    
    def get_best_estimate(self):
        """
        获取最佳估算值
        
        综合多种方法的结果
        
        Returns:
            float: 赛道长度的最佳估算
        """
        estimates = []
        
        # 方法1：从完成的圈数估算（最准确）
        if self.distance_samples:
            estimates.append(self.estimate_from_completed_laps())
        
        # 方法2：从位置样本估算
        if len(self.position_samples) >= 3:
            try:
                estimates.append(self.estimate_from_samples())
            except:
                pass
        
        # 方法3：从已知半径估算
        if self.radius:
            estimates.append(self.estimate_from_radius())
        
        if estimates:
            # 返回中位数（对异常值更鲁棒）
            return np.median(estimates)
        
        return 0.0


# ==================== 使用示例 ====================

def example_simple_lap_detection():
    """
    示例1：简单的圈数检测
    """
    # 起点位置
    start_position = [0, -1.2, 0]
    
    # 创建检测器
    lap_detector = LapDetector(start_position, threshold=0.3)
    
    # 模拟机器人运动
    positions = [
        [0, -1.2, 0],    # 起点
        [0.5, -1.0, 0],
        [1.0, 0, 0],
        [0.5, 1.0, 0],
        [0, 1.2, 0],
        [-0.5, 1.0, 0],
        [-1.0, 0, 0],
        [-0.5, -1.0, 0],
        [0, -1.2, 0],    # 回到起点
    ]
    
    for i in range(1, len(positions)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        
        # 计算移动距离
        distance = np.linalg.norm(
            np.array(curr_pos[:2]) - np.array(prev_pos[:2])
        )
        
        # 更新检测器
        completed = lap_detector.update(curr_pos, distance)
        
        if completed:
            print(f"完成一圈！圈长: {lap_detector.get_circuit_length():.2f}m")


def example_sector_lap_detection():
    """
    示例2：使用扇区检测（更可靠）
    """
    start_position = [0, -1.2, 0]
    center_position = [0, 0, 0]
    
    # 创建扇区检测器（4个扇区）
    lap_detector = SectorLapDetector(
        start_position, 
        center_position, 
        num_sectors=4,
        threshold=0.3
    )
    
    # 模拟运动...
    # （代码类似上面的例子）
    
    print(f"当前进度: {lap_detector.get_progress() * 100:.1f}%")


def example_circuit_length_estimation():
    """
    示例3：估算赛道长度
    """
    # 方法1：从已知半径估算
    estimator = CircuitLengthEstimator(center=[0, 0], radius=1.2)
    length1 = estimator.estimate_from_radius()
    print(f"从半径估算: {length1:.2f}m")
    
    # 方法2：从完成的圈数估算
    estimator.add_distance_sample(7.54)  # 第1圈
    estimator.add_distance_sample(7.52)  # 第2圈
    estimator.add_distance_sample(7.53)  # 第3圈
    length2 = estimator.estimate_from_completed_laps()
    print(f"从完成圈估算: {length2:.2f}m")
    
    # 方法3：综合估算
    length3 = estimator.get_best_estimate()
    print(f"最佳估算: {length3:.2f}m")


def example_integration_with_fitness():
    """
    示例4：与适应度评估器集成
    """
    from fitness_evaluator import FitnessEvaluator
    
    # 创建适应度评估器和圈数检测器
    fitness_eval = FitnessEvaluator()
    lap_detector = SectorLapDetector(
        start_position=[0, -1.2, 0],
        center_position=[0, 0, 0],
        num_sectors=4
    )
    
    # 在控制循环中
    while True:
        # ... 获取传感器数据和位置 ...
        sensor_data = {}
        motor_speeds = [0.5, 0.5]
        position = [0, 0, 0]
        distance_moved = 0.1
        
        # 更新适应度评估
        fitness_eval.update(sensor_data, motor_speeds, position)
        
        # 更新圈数检测
        if lap_detector.update(position, distance_moved):
            print(f"完成一圈！")
            print(f"圈长: {lap_detector.get_circuit_length():.2f}m")
            print(f"适应度: {fitness_eval.calculate_fitness():.2f}")
            
            # 标记完成一圈
            fitness_eval.completed_lap = True
        
        # 显示进度
        progress = lap_detector.get_progress()
        print(f"进度: {progress * 100:.1f}%")


if __name__ == '__main__':
    print("圈数检测示例\n")
    
    print("=" * 60)
    print("示例1: 简单圈数检测")
    print("=" * 60)
    example_simple_lap_detection()
    
    print("\n" + "=" * 60)
    print("示例3: 赛道长度估算")
    print("=" * 60)
    example_circuit_length_estimation()
