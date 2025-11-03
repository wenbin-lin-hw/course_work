"""
机器人控制器 - Webots e-puck机器人的控制接口

这个类负责：
1. 与Webots仿真环境交互
2. 读取传感器数据
3. 控制电机
4. 使用神经网络做决策
"""
import numpy as np
from controller import Robot, Motor, DistanceSensor, GPS
from neural_network import NeuralNetwork
from fitness_evaluator import FitnessEvaluator
from config import ROBOT_CONFIG


class EPuckController:
    """
    E-puck机器人控制器

    封装了与Webots的所有交互
    """

    def __init__(self, neural_network=None):
        """
        初始化控制器

        Args:
            neural_network: 神经网络（机器人的"大脑"）
        """
        # 初始化Webots机器人
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # 配置参数
        self.config = ROBOT_CONFIG
        self.max_speed = self.config['max_speed']

        # 神经网络
        self.neural_network = neural_network

        # 初始化设备
        self._init_motors()
        self._init_sensors()

        # 适应度评估器
        self.fitness_evaluator = FitnessEvaluator()

        # 起始位置（用于检测完成一圈）
        self.start_position = None

        print("机器人控制器初始化完成")

    def _init_motors(self):
        """初始化电机"""
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')

        # 设置电机为速度控制模式
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # 初始速度为0
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def _init_sensors(self):
        """初始化传感器"""
        # 距离传感器（用于避障）
        self.distance_sensors = []
        for name in self.config['distance_sensor_names']:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.distance_sensors.append(sensor)

        # 地面传感器（用于循迹）
        self.ground_sensors = []
        for name in self.config['ground_sensor_names']:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.ground_sensors.append(sensor)

        # GPS（用于获取位置）
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)
        else:
            print("警告: GPS设备未找到，将无法追踪位置")

    def read_sensors(self):
        """
        读取所有传感器数据

        Returns:
            dict: 包含所有传感器数据的字典
        """
        # 读取距离传感器
        distance_values = [sensor.getValue() for sensor in self.distance_sensors]

        # 读取地面传感器
        ground_values = [sensor.getValue() for sensor in self.ground_sensors]

        # 读取GPS位置
        position = None
        if self.gps:
            position = self.gps.getValues()

        return {
            'distance_sensors': distance_values,
            'ground_sensors': ground_values,
            'position': position
        }

    def normalize_sensors(self, sensor_data):
        """
        归一化传感器数据到[0, 1]范围

        Args:
            sensor_data: 原始传感器数据

        Returns:
            numpy数组: 归一化后的传感器数据
        """
        # 归一化距离传感器 [0, 4096] -> [0, 1]
        distance_normalized = np.array(sensor_data['distance_sensors']) / 4096.0

        # 归一化地面传感器 [0, 1000] -> [0, 1]
        ground_normalized = np.array(sensor_data['ground_sensors']) / 1000.0

        # 合并所有传感器数据
        all_sensors = np.concatenate([distance_normalized, ground_normalized])

        return all_sensors

    def set_motor_speeds(self, left_speed, right_speed):
        """
        设置电机速度

        Args:
            left_speed: 左轮速度 [-1, 1]
            right_speed: 右轮速度 [-1, 1]
        """
        # 将[-1, 1]映射到[-max_speed, max_speed]
        left_velocity = left_speed * self.max_speed
        right_velocity = right_speed * self.max_speed

        self.left_motor.setVelocity(left_velocity)
        self.right_motor.setVelocity(right_velocity)

    def step(self):
        """
        执行一个控制步骤

        1. 读取传感器
        2. 使用神经网络计算控制指令
        3. 执行控制指令
        4. 更新适应度评估

        Returns:
            bool: 是否成功执行（False表示仿真结束）
        """
        # 执行一个仿真步
        if self.robot.step(self.timestep) == -1:
            return False

        # 读取传感器
        sensor_data = self.read_sensors()

        # 记录起始位置
        if self.start_position is None and sensor_data['position']:
            self.start_position = sensor_data['position']

        # 归一化传感器数据
        normalized_sensors = self.normalize_sensors(sensor_data)

        # 使用神经网络计算控制指令
        if self.neural_network:
            outputs = self.neural_network.forward(normalized_sensors)
            left_speed = outputs[0]
            right_speed = outputs[1]
        else:
            # 如果没有神经网络，使用默认行为（直行）
            left_speed = 0.5
            right_speed = 0.5

        # 设置电机速度
        self.set_motor_speeds(left_speed, right_speed)

        # 更新适应度评估
        if sensor_data['position']:
            self.fitness_evaluator.update(
                sensor_data,
                [left_speed, right_speed],
                sensor_data['position']
            )

            # 检查是否完成一圈
            if self.start_position:
                self.fitness_evaluator.check_lap_completion(
                    sensor_data['position'],
                    self.start_position
                )

        return True

    def run(self, duration):
        """
        运行机器人指定时间

        Args:
            duration: 运行时间（秒）

        Returns:
            float: 适应度分数
        """
        self.fitness_evaluator.reset()
        self.start_position = None

        steps = int(duration * 1000 / self.timestep)

        for i in range(steps):
            if not self.step():
                break

        return self.fitness_evaluator.calculate_fitness()

    def get_fitness(self):
        """
        获取当前适应度分数

        Returns:
            float: 适应度分数
        """
        return self.fitness_evaluator.calculate_fitness()

    def get_stats(self):
        """
        获取详细统计信息

        Returns:
            dict: 统计信息
        """
        return self.fitness_evaluator.get_stats()

    def reset(self):
        """重置控制器状态"""
        self.fitness_evaluator.reset()
        self.start_position = None
        self.set_motor_speeds(0.0, 0.0)

    def set_neural_network(self, neural_network):
        """
        设置神经网络

        Args:
            neural_network: 新的神经网络
        """
        self.neural_network = neural_network

    def get_position(self):
        """
        获取当前位置

        Returns:
            list: [x, y, z] 位置坐标
        """
        if self.gps:
            return self.gps.getValues()
        return None

    def get_lap_time(self):
        """
        获取跑一圈的时间

        Returns:
            float: 时间（秒），如果未完成返回None
        """
        if self.fitness_evaluator.completed_lap:
            return self.fitness_evaluator.step_count * self.timestep / 1000.0
        return None
"""
机器人控制器 - Webots e-puck机器人的控制接口

这个类负责：
1. 与Webots仿真环境交互
2. 读取传感器数据
3. 控制电机
4. 使用神经网络做决策
"""
import numpy as np
from controller import Robot, Motor, DistanceSensor, GPS
from neural_network import NeuralNetwork
from fitness_evaluator import FitnessEvaluator
from config import ROBOT_CONFIG


class EPuckController:
    """
    E-puck机器人控制器
    
    封装了与Webots的所有交互
    """
    
    def __init__(self, neural_network=None):
        """
        初始化控制器
        
        Args:
            neural_network: 神经网络（机器人的"大脑"）
        """
        # 初始化Webots机器人
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # 配置参数
        self.config = ROBOT_CONFIG
        self.max_speed = self.config['max_speed']
        
        # 神经网络
        self.neural_network = neural_network
        
        # 初始化设备
        self._init_motors()
        self._init_sensors()
        
        # 适应度评估器
        self.fitness_evaluator = FitnessEvaluator()
        
        # 起始位置（用于检测完成一圈）
        self.start_position = None
        
        print("机器人控制器初始化完成")
    
    def _init_motors(self):
        """初始化电机"""
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        
        # 设置电机为速度控制模式
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # 初始速度为0
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
    
    def _init_sensors(self):
        """初始化传感器"""
        # 距离传感器（用于避障）
        self.distance_sensors = []
        for name in self.config['distance_sensor_names']:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.distance_sensors.append(sensor)
        
        # 地面传感器（用于循迹）
        self.ground_sensors = []
        for name in self.config['ground_sensor_names']:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.ground_sensors.append(sensor)
        
        # GPS（用于获取位置）
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)
        else:
            print("警告: GPS设备未找到，将无法追踪位置")
    
    def read_sensors(self):
        """
        读取所有传感器数据
        
        Returns:
            dict: 包含所有传感器数据的字典
        """
        # 读取距离传感器
        distance_values = [sensor.getValue() for sensor in self.distance_sensors]
        
        # 读取地面传感器
        ground_values = [sensor.getValue() for sensor in self.ground_sensors]
        
        # 读取GPS位置
        position = None
        if self.gps:
            position = self.gps.getValues()
        
        return {
            'distance_sensors': distance_values,
            'ground_sensors': ground_values,
            'position': position
        }
    
    def normalize_sensors(self, sensor_data):
        """
        归一化传感器数据到[0, 1]范围
        
        Args:
            sensor_data: 原始传感器数据
            
        Returns:
            numpy数组: 归一化后的传感器数据
        """
        # 归一化距离传感器 [0, 4096] -> [0, 1]
        distance_normalized = np.array(sensor_data['distance_sensors']) / 4096.0
        
        # 归一化地面传感器 [0, 1000] -> [0, 1]
        ground_normalized = np.array(sensor_data['ground_sensors']) / 1000.0
        
        # 合并所有传感器数据
        all_sensors = np.concatenate([distance_normalized, ground_normalized])
        
        return all_sensors
    
    def set_motor_speeds(self, left_speed, right_speed):
        """
        设置电机速度
        
        Args:
            left_speed: 左轮速度 [-1, 1]
            right_speed: 右轮速度 [-1, 1]
        """
        # 将[-1, 1]映射到[-max_speed, max_speed]
        left_velocity = left_speed * self.max_speed
        right_velocity = right_speed * self.max_speed
        
        self.left_motor.setVelocity(left_velocity)
        self.right_motor.setVelocity(right_velocity)
    
    def step(self):
        """
        执行一个控制步骤
        
        1. 读取传感器
        2. 使用神经网络计算控制指令
        3. 执行控制指令
        4. 更新适应度评估
        
        Returns:
            bool: 是否成功执行（False表示仿真结束）
        """
        # 执行一个仿真步
        if self.robot.step(self.timestep) == -1:
            return False
        
        # 读取传感器
        sensor_data = self.read_sensors()
        
        # 记录起始位置
        if self.start_position is None and sensor_data['position']:
            self.start_position = sensor_data['position']
        
        # 归一化传感器数据
        normalized_sensors = self.normalize_sensors(sensor_data)
        
        # 使用神经网络计算控制指令
        if self.neural_network:
            outputs = self.neural_network.forward(normalized_sensors)
            left_speed = outputs[0]
            right_speed = outputs[1]
        else:
            # 如果没有神经网络，使用默认行为（直行）
            left_speed = 0.5
            right_speed = 0.5
        
        # 设置电机速度
        self.set_motor_speeds(left_speed, right_speed)
        
        # 更新适应度评估
        if sensor_data['position']:
            self.fitness_evaluator.update(
                sensor_data,
                [left_speed, right_speed],
                sensor_data['position']
            )
            
            # 检查是否完成一圈
            if self.start_position:
                self.fitness_evaluator.check_lap_completion(
                    sensor_data['position'],
                    self.start_position
                )
        
        return True
    
    def run(self, duration):
        """
        运行机器人指定时间
        
        Args:
            duration: 运行时间（秒）
            
        Returns:
            float: 适应度分数
        """
        self.fitness_evaluator.reset()
        self.start_position = None
        
        steps = int(duration * 1000 / self.timestep)
        
        for i in range(steps):
            if not self.step():
                break
        
        return self.fitness_evaluator.calculate_fitness()
    
    def get_fitness(self):
        """
        获取当前适应度分数
        
        Returns:
            float: 适应度分数
        """
        return self.fitness_evaluator.calculate_fitness()
    
    def get_stats(self):
        """
        获取详细统计信息
        
        Returns:
            dict: 统计信息
        """
        return self.fitness_evaluator.get_stats()
    
    def reset(self):
        """重置控制器状态"""
        self.fitness_evaluator.reset()
        self.start_position = None
        self.set_motor_speeds(0.0, 0.0)
    
    def set_neural_network(self, neural_network):
        """
        设置神经网络
        
        Args:
            neural_network: 新的神经网络
        """
        self.neural_network = neural_network
    
    def get_position(self):
        """
        获取当前位置
        
        Returns:
            list: [x, y, z] 位置坐标
        """
        if self.gps:
            return self.gps.getValues()
        return None
    
    def get_lap_time(self):
        """
        获取跑一圈的时间
        
        Returns:
            float: 时间（秒），如果未完成返回None
        """
        if self.fitness_evaluator.completed_lap:
            return self.fitness_evaluator.step_count * self.timestep / 1000.0
        return None
