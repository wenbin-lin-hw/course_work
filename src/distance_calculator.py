"""
距离计算模块 - 多种方法计算机器人行驶距离

提供三种方法计算距离：
1. GPS方法（如果有GPS传感器）
2. Supervisor方法（使用Supervisor获取位置）- 推荐！
3. 编码器方法（使用轮子编码器估算）

当E-puck没有GPS时，可以使用方法2或3
"""
import numpy as np
from config import ROBOT_CONFIG


class DistanceCalculator:
    """
    距离计算器基类
    """
    
    def __init__(self):
        self.total_distance = 0.0
        self.prev_position = None
    
    def reset(self):
        """重置距离计算器"""
        self.total_distance = 0.0
        self.prev_position = None
    
    def update(self, position):
        """
        更新距离
        
        Args:
            position: [x, y, z] 位置坐标
            
        Returns:
            float: 本次移动的距离
        """
        if position is None:
            return 0.0
        
        if self.prev_position is None:
            self.prev_position = position
            return 0.0
        
        # 计算欧氏距离（只考虑x和y）
        distance = np.sqrt(
            (position[0] - self.prev_position[0]) ** 2 +
            (position[1] - self.prev_position[1]) ** 2
        )
        
        self.total_distance += distance
        self.prev_position = position
        
        return distance
    
    def get_total_distance(self):
        """获取总距离"""
        return self.total_distance


class GPSDistanceCalculator(DistanceCalculator):
    """
    基于GPS的距离计算器
    
    使用E-puck自带的GPS传感器
    优点：准确
    缺点：需要GPS设备
    """
    
    def __init__(self, gps_device):
        """
        初始化
        
        Args:
            gps_device: GPS设备对象
        """
        super().__init__()
        self.gps = gps_device
    
    def get_position(self):
        """获取当前位置"""
        if self.gps:
            return self.gps.getValues()
        return None
    
    def update_from_gps(self):
        """从GPS更新距离"""
        position = self.get_position()
        return self.update(position)


class SupervisorDistanceCalculator(DistanceCalculator):
    """
    基于Supervisor的距离计算器
    
    使用Supervisor获取机器人位置
    优点：不需要GPS，非常准确
    缺点：需要Supervisor权限
    
    这是推荐的方法！
    """
    
    def __init__(self, robot_node):
        """
        初始化
        
        Args:
            robot_node: Supervisor获取的机器人节点
        """
        super().__init__()
        self.robot_node = robot_node
    
    def get_position(self):
        """
        从Supervisor获取机器人位置
        
        Returns:
            [x, y, z] 或 None
        """
        if self.robot_node:
            translation_field = self.robot_node.getField('translation')
            if translation_field:
                return translation_field.getSFVec3f()
        return None
    
    def update_from_supervisor(self):
        """从Supervisor更新距离"""
        position = self.get_position()
        return self.update(position)


class EncoderDistanceCalculator:
    """
    基于轮子编码器的距离计算器
    
    通过轮子旋转角度估算行驶距离
    优点：不需要GPS或Supervisor
    缺点：有累积误差，打滑时不准确
    
    原理：
    distance = (left_wheel_distance + right_wheel_distance) / 2
    wheel_distance = wheel_radius × rotation_angle
    """
    
    def __init__(self, left_sensor, right_sensor, wheel_radius=None):
        """
        初始化
        
        Args:
            left_sensor: 左轮位置传感器
            right_sensor: 右轮位置传感器
            wheel_radius: 轮子半径（米）
        """
        self.left_sensor = left_sensor
        self.right_sensor = right_sensor
        
        if wheel_radius is None:
            wheel_radius = ROBOT_CONFIG['wheel_radius']
        self.wheel_radius = wheel_radius
        
        self.total_distance = 0.0
        self.prev_left_position = None
        self.prev_right_position = None
    
    def reset(self):
        """重置距离计算器"""
        self.total_distance = 0.0
        self.prev_left_position = None
        self.prev_right_position = None
    
    def update(self):
        """
        从编码器更新距离
        
        Returns:
            float: 本次移动的距离
        """
        # 读取当前轮子位置（弧度）
        left_position = self.left_sensor.getValue()
        right_position = self.right_sensor.getValue()
        
        # 第一次调用，只记录位置
        if self.prev_left_position is None:
            self.prev_left_position = left_position
            self.prev_right_position = right_position
            return 0.0
        
        # 计算轮子旋转角度
        left_delta = left_position - self.prev_left_position
        right_delta = right_position - self.prev_right_position
        
        # 计算每个轮子移动的距离
        left_distance = abs(left_delta) * self.wheel_radius
        right_distance = abs(right_delta) * self.wheel_radius
        
        # 机器人移动距离 = 两轮平均距离
        distance = (left_distance + right_distance) / 2.0
        
        self.total_distance += distance
        
        # 更新上一次位置
        self.prev_left_position = left_position
        self.prev_right_position = right_position
        
        return distance
    
    def get_total_distance(self):
        """获取总距离"""
        return self.total_distance
    
    def get_position_estimate(self, initial_position=[0, 0, 0], initial_angle=0):
        """
        估算机器人位置（可选功能）
        
        通过轮子编码器估算机器人的位置和朝向
        这是一个简化的里程计算法
        
        Args:
            initial_position: 初始位置 [x, y, z]
            initial_angle: 初始角度（弧度）
            
        Returns:
            [x, y, z, angle]: 估算的位置和角度
        """
        if self.prev_left_position is None:
            return initial_position + [initial_angle]
        
        # 读取当前轮子位置
        left_position = self.left_sensor.getValue()
        right_position = self.right_sensor.getValue()
        
        # 计算轮子旋转角度
        left_delta = left_position - self.prev_left_position
        right_delta = right_position - self.prev_right_position
        
        # 计算每个轮子移动的距离
        left_distance = left_delta * self.wheel_radius
        right_distance = right_delta * self.wheel_radius
        
        # 计算机器人移动和旋转
        distance = (left_distance + right_distance) / 2.0
        angle_delta = (right_distance - left_distance) / ROBOT_CONFIG['axle_length']
        
        # 更新位置和角度
        new_angle = initial_angle + angle_delta
        new_x = initial_position[0] + distance * np.cos(new_angle)
        new_y = initial_position[1] + distance * np.sin(new_angle)
        new_z = initial_position[2]
        
        return [new_x, new_y, new_z, new_angle]


class HybridDistanceCalculator:
    """
    混合距离计算器
    
    优先使用GPS，如果不可用则使用Supervisor，
    如果都不可用则使用编码器
    
    这是最灵活的方法！
    """
    
    def __init__(self, gps_device=None, robot_node=None, 
                 left_sensor=None, right_sensor=None):
        """
        初始化
        
        Args:
            gps_device: GPS设备（可选）
            robot_node: Supervisor机器人节点（可选）
            left_sensor: 左轮编码器（可选）
            right_sensor: 右轮编码器（可选）
        """
        self.calculators = []
        self.active_calculator = None
        
        # 尝试初始化各种计算器
        if gps_device:
            try:
                calc = GPSDistanceCalculator(gps_device)
                self.calculators.append(('GPS', calc))
                print("  距离计算: GPS可用")
            except:
                pass
        
        if robot_node:
            try:
                calc = SupervisorDistanceCalculator(robot_node)
                self.calculators.append(('Supervisor', calc))
                print("  距离计算: Supervisor可用")
            except:
                pass
        
        if left_sensor and right_sensor:
            try:
                calc = EncoderDistanceCalculator(left_sensor, right_sensor)
                self.calculators.append(('Encoder', calc))
                print("  距离计算: 编码器可用")
            except:
                pass
        
        # 选择第一个可用的计算器
        if self.calculators:
            self.active_calculator = self.calculators[0]
            print(f"  使用{self.active_calculator[0]}计算距离")
        else:
            print("  警告: 没有可用的距离计算方法！")
    
    def reset(self):
        """重置所有计算器"""
        for name, calc in self.calculators:
            calc.reset()
    
    def update(self):
        """更新距离"""
        if self.active_calculator:
            name, calc = self.active_calculator
            
            if name == 'GPS':
                return calc.update_from_gps()
            elif name == 'Supervisor':
                return calc.update_from_supervisor()
            elif name == 'Encoder':
                return calc.update()
        
        return 0.0
    
    def get_total_distance(self):
        """获取总距离"""
        if self.active_calculator:
            return self.active_calculator[1].get_total_distance()
        return 0.0
    
    def get_position(self):
        """获取当前位置（如果可用）"""
        if self.active_calculator:
            name, calc = self.active_calculator
            if hasattr(calc, 'get_position'):
                return calc.get_position()
        return None


# ==================== 使用示例 ====================

def example_usage_with_supervisor():
    """
    示例：在Supervisor控制器中使用
    """
    from controller import Supervisor
    
    supervisor = Supervisor()
    
    # 获取机器人节点
    robot_node = supervisor.getFromDef('EPUCK_0')
    
    # 创建距离计算器
    distance_calc = SupervisorDistanceCalculator(robot_node)
    
    # 在控制循环中
    while supervisor.step(32) != -1:
        # 更新距离
        distance_calc.update_from_supervisor()
        
        # 获取总距离
        total_distance = distance_calc.get_total_distance()
        print(f"总距离: {total_distance:.3f}m")


def example_usage_with_encoder():
    """
    示例：在普通机器人控制器中使用编码器
    """
    from controller import Robot
    
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # 获取轮子位置传感器
    left_sensor = robot.getDevice('left wheel sensor')
    right_sensor = robot.getDevice('right wheel sensor')
    left_sensor.enable(timestep)
    right_sensor.enable(timestep)
    
    # 创建距离计算器
    distance_calc = EncoderDistanceCalculator(left_sensor, right_sensor)
    
    # 在控制循环中
    while robot.step(timestep) != -1:
        # 更新距离
        distance_calc.update()
        
        # 获取总距离
        total_distance = distance_calc.get_total_distance()
        print(f"总距离: {total_distance:.3f}m")


def example_usage_hybrid():
    """
    示例：使用混合计算器（自动选择最佳方法）
    """
    from controller import Robot
    
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # 尝试获取各种设备
    gps = robot.getDevice('gps')
    if gps:
        gps.enable(timestep)
    
    left_sensor = robot.getDevice('left wheel sensor')
    right_sensor = robot.getDevice('right wheel sensor')
    if left_sensor:
        left_sensor.enable(timestep)
    if right_sensor:
        right_sensor.enable(timestep)
    
    # 创建混合计算器（自动选择可用的方法）
    distance_calc = HybridDistanceCalculator(
        gps_device=gps,
        left_sensor=left_sensor,
        right_sensor=right_sensor
    )
    
    # 在控制循环中
    while robot.step(timestep) != -1:
        # 更新距离
        distance_calc.update()
        
        # 获取总距离
        total_distance = distance_calc.get_total_distance()
        print(f"总距离: {total_distance:.3f}m")
