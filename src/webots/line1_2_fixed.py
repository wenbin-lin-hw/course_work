"""
E-puck Line Following with Obstacle Avoidance - FIXED VERSION

主要修复：
1. 修正了距离传感器索引映射问题
2. 添加了详细的调试信息
3. 降低了障碍物检测阈值，使其更敏感
4. 添加了传感器值打印功能
"""
from controller import Robot
from datetime import datetime
import math
import numpy as np


class Controller:
    def __init__(self, robot):
        # Robot Parameters
        self.robot = robot
        self.time_step = 32  # ms
        self.max_speed = 1  # m/s

        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0

        # Enable Proximity Sensors
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)

        # Enable Ground Sensors
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)

        # Data
        self.inputs = []
        self.inputsPrevious = []
        
        # Distance sensor values (raw)
        self.distance_sensors_raw = [0] * 8

        # Flag
        self.flag_turn = 0

        # ========== Obstacle Avoidance State Machine ==========
        self.state = 'LINE_FOLLOWING'
        self.obstacle_side = None
        self.avoidance_counter = 0
        self.search_counter = 0

        # ========== FIXED: 降低阈值，使障碍物检测更敏感 ==========
        self.OBSTACLE_THRESHOLD = 0.05  # 从0.15降到0.05 (更敏感)
        self.OBSTACLE_CLOSE_THRESHOLD = 0.15  # 从0.3降到0.15

        # Avoidance parameters
        self.AVOIDANCE_DURATION = 60  # 增加避障时间
        self.SEARCH_DURATION = 120
        self.PARALLEL_DURATION = 30

        # Debug mode
        self.DEBUG = True  # 设置为True以查看调试信息
        self.debug_counter = 0

        print("=" * 60)
        print("E-puck Controller with Obstacle Avoidance - FIXED VERSION")
        print("=" * 60)
        print(f"Obstacle threshold: {self.OBSTACLE_THRESHOLD}")
        print(f"Debug mode: {self.DEBUG}")
        print("=" * 60)

    def clip_value(self, value, min_max):
        if value > min_max:
            return min_max
        elif value < -min_max:
            return -min_max
        return value

    def print_sensor_debug(self):
        """
        打印传感器调试信息
        """
        if not self.DEBUG:
            return
        
        self.debug_counter += 1
        if self.debug_counter % 30 != 0:  # 每30步打印一次
            return
        
        print("\n" + "=" * 60)
        print(f"State: {self.state}")
        print("-" * 60)
        
        # 打印地面传感器
        print("Ground Sensors (normalized):")
        print(f"  Left:   {self.inputs[0]:.3f}")
        print(f"  Center: {self.inputs[1]:.3f}")
        print(f"  Right:  {self.inputs[2]:.3f}")
        
        # 打印距离传感器（原始值）
        print("\nDistance Sensors (raw values):")
        print(f"  ps0 (front-right):  {self.distance_sensors_raw[0]:.1f}")
        print(f"  ps1 (front-right):  {self.distance_sensors_raw[1]:.1f}")
        print(f"  ps2 (right-side):   {self.distance_sensors_raw[2]:.1f}")
        print(f"  ps3 (back-right):   {self.distance_sensors_raw[3]:.1f}")
        print(f"  ps4 (back-left):    {self.distance_sensors_raw[4]:.1f}")
        print(f"  ps5 (left-side):    {self.distance_sensors_raw[5]:.1f}")
        print(f"  ps6 (front-left):   {self.distance_sensors_raw[6]:.1f}")
        print(f"  ps7 (front-left):   {self.distance_sensors_raw[7]:.1f}")
        
        # 打印距离传感器（归一化值）
        if len(self.inputs) >= 9:
            print("\nDistance Sensors (normalized, in inputs array):")
            print(f"  inputs[3] (ps0): {self.inputs[3]:.3f}")
            print(f"  inputs[4] (ps1): {self.inputs[4]:.3f}")
            print(f"  inputs[5] (ps2): {self.inputs[5]:.3f}")
            print(f"  inputs[6] (ps5): {self.inputs[6]:.3f}")
            print(f"  inputs[7] (ps6): {self.inputs[7]:.3f}")
            print(f"  inputs[8] (ps7): {self.inputs[8]:.3f}")
        
        print("=" * 60)

    def detect_obstacle(self):
        """
        检测障碍物 - FIXED VERSION
        
        E-puck传感器布局：
                ps7  ps6
                  \\//
                   ||
        ps5 ----  ROBOT  ---- ps2
                   ||
                  //\\
                ps0  ps1
        
        inputs数组映射：
        inputs[0-2]: 地面传感器 (gs0, gs1, gs2)
        inputs[3]: ps0 (前右)
        inputs[4]: ps1 (前右)
        inputs[5]: ps2 (右侧)
        inputs[6]: ps5 (左侧)
        inputs[7]: ps6 (前左)
        inputs[8]: ps7 (前左)
        
        Returns:
            tuple: (has_obstacle, obstacle_side, obstacle_distance)
        """
        if len(self.inputs) < 9:
            return False, None, 0
        
        # ========== FIXED: 正确的索引映射 ==========
        # 前方传感器
        ps0 = self.inputs[3]  # 前右
        ps1 = self.inputs[4]  # 前右
        ps6 = self.inputs[7]  # 前左
        ps7 = self.inputs[8]  # 前左
        
        # 侧面传感器
        ps2 = self.inputs[5]  # 右侧
        ps5 = self.inputs[6]  # 左侧
        
        # 计算前方左右的最大值
        front_left = max(ps6, ps7)
        front_right = max(ps0, ps1)
        front_max = max(front_left, front_right)
        
        # 检测障碍物
        has_obstacle = False
        obstacle_side = None
        obstacle_distance = 0
        
        # 1. 检查前方障碍物（最高优先级）
        if front_max > self.OBSTACLE_THRESHOLD:
            has_obstacle = True
            obstacle_distance = front_max
            
            if self.DEBUG and self.debug_counter % 10 == 0:
                print(f"\n!!! FRONT OBSTACLE DETECTED !!!")
                print(f"  front_left={front_left:.3f}, front_right={front_right:.3f}")
            
            # 判断障碍物在哪一侧
            if front_left > front_right:
                obstacle_side = 'front_left'
            else:
                obstacle_side = 'front_right'
        
        # 2. 检查左侧障碍物
        elif ps5 > self.OBSTACLE_THRESHOLD:
            has_obstacle = True
            obstacle_side = 'left'
            obstacle_distance = ps5
            
            if self.DEBUG and self.debug_counter % 10 == 0:
                print(f"\n!!! LEFT OBSTACLE DETECTED !!! (ps5={ps5:.3f})")
        
        # 3. 检查右侧障碍物
        elif ps2 > self.OBSTACLE_THRESHOLD:
            has_obstacle = True
            obstacle_side = 'right'
            obstacle_distance = ps2
            
            if self.DEBUG and self.debug_counter % 10 == 0:
                print(f"\n!!! RIGHT OBSTACLE DETECTED !!! (ps2={ps2:.3f})")
        
        return has_obstacle, obstacle_side, obstacle_distance

    def is_on_line(self):
        """
        检查机器人是否在黑线上
        """
        if len(self.inputs) < 3:
            return False
        
        line_threshold = 0.5
        on_line = (self.inputs[0] < line_threshold or
                   self.inputs[1] < line_threshold or
                   self.inputs[2] < line_threshold)
        
        return on_line

    def obstacle_avoidance_behavior(self):
        """
        障碍物避障状态机
        """
        # 检测障碍物
        has_obstacle, obstacle_side, obstacle_distance = self.detect_obstacle()

        # ===== State 1: LINE_FOLLOWING =====
        if self.state == 'LINE_FOLLOWING':
            if has_obstacle:
                self.state = 'OBSTACLE_DETECTED'
                self.obstacle_side = obstacle_side
                self.avoidance_counter = 0
                print(f"\n{'='*60}")
                print(f"[{self.state}] Obstacle detected!")
                print(f"  Side: {obstacle_side}")
                print(f"  Distance: {obstacle_distance:.3f}")
                print(f"{'='*60}")
                return 0, 0  # 短暂停止
            else:
                return self.line_following_behavior()

        # ===== State 2: OBSTACLE_DETECTED =====
        elif self.state == 'OBSTACLE_DETECTED':
            if obstacle_side in ['front_left', 'left']:
                self.state = 'AVOIDING_RIGHT'
                print(f"[{self.state}] Decision: Avoiding RIGHT")
            else:
                self.state = 'AVOIDING_LEFT'
                print(f"[{self.state}] Decision: Avoiding LEFT")

            self.avoidance_counter = 0
            return 0, 0

        # ===== State 3: AVOIDING_LEFT =====
        elif self.state == 'AVOIDING_LEFT':
            self.avoidance_counter += 1

            # 阶段1: 急转左 (前1/3)
            if self.avoidance_counter < self.AVOIDANCE_DURATION // 3:
                left_vel = 0.1
                right_vel = 0.9
            # 阶段2: 向前偏左 (中间1/3)
            elif self.avoidance_counter < 2 * self.AVOIDANCE_DURATION // 3:
                left_vel = 0.7
                right_vel = 0.9
            # 阶段3: 转回右侧 (后1/3)
            else:
                left_vel = 0.9
                right_vel = 0.5

            if self.avoidance_counter >= self.AVOIDANCE_DURATION:
                self.state = 'SEARCHING_LINE'
                self.search_counter = 0
                print(f"[{self.state}] Avoidance complete, searching for line...")

            # 如果仍然检测到障碍物，继续避障
            if has_obstacle and obstacle_distance > self.OBSTACLE_CLOSE_THRESHOLD:
                self.avoidance_counter = max(0, self.avoidance_counter - 5)

            return left_vel, right_vel

        # ===== State 4: AVOIDING_RIGHT =====
        elif self.state == 'AVOIDING_RIGHT':
            self.avoidance_counter += 1

            # 阶段1: 急转右 (前1/3)
            if self.avoidance_counter < self.AVOIDANCE_DURATION // 3:
                left_vel = 0.9
                right_vel = 0.1
            # 阶段2: 向前偏右 (中间1/3)
            elif self.avoidance_counter < 2 * self.AVOIDANCE_DURATION // 3:
                left_vel = 0.9
                right_vel = 0.7
            # 阶段3: 转回左侧 (后1/3)
            else:
                left_vel = 0.5
                right_vel = 0.9

            if self.avoidance_counter >= self.AVOIDANCE_DURATION:
                self.state = 'SEARCHING_LINE'
                self.search_counter = 0
                print(f"[{self.state}] Avoidance complete, searching for line...")

            # 如果仍然检测到障碍物，继续避障
            if has_obstacle and obstacle_distance > self.OBSTACLE_CLOSE_THRESHOLD:
                self.avoidance_counter = max(0, self.avoidance_counter - 5)

            return left_vel, right_vel

        # ===== State 5: SEARCHING_LINE =====
        elif self.state == 'SEARCHING_LINE':
            self.search_counter += 1

            # 检查是否找到黑线
            if self.is_on_line():
                self.state = 'LINE_FOLLOWING'
                print(f"[{self.state}] Line found! Resuming line following")
                print(f"{'='*60}\n")
                return self.line_following_behavior()

            # 继续搜索
            if self.obstacle_side in ['front_left', 'left']:
                # 向右转寻找线
                left_vel = 0.8
                right_vel = 0.4
            else:
                # 向左转寻找线
                left_vel = 0.4
                right_vel = 0.8

            # 超时
            if self.search_counter >= self.SEARCH_DURATION:
                self.state = 'LINE_FOLLOWING'
                print(f"[{self.state}] Search timeout, resuming line following")
                print(f"{'='*60}\n")

            return left_vel, right_vel

        # 默认：返回循迹行为
        return self.line_following_behavior()

    def line_following_behavior(self):
        """
        循迹行为
        """
        # 转弯标志
        if self.flag_turn:
            left_vel = -0.3
            right_vel = 0.3
            if np.min(self.inputs[0:3]) < 0.35:
                self.flag_turn = 0
            return left_vel, right_vel

        # 检查线的末端
        if len(self.inputsPrevious) >= 3:
            if (np.min(self.inputs[0:3]) - np.min(self.inputsPrevious[0:3])) > 0.2:
                self.flag_turn = 1
                return -0.3, 0.3

        # 根据地面传感器循迹
        if self.inputs[0] < self.inputs[1] and self.inputs[0] < self.inputs[2]:
            # 线在左边，左转
            left_vel = 0.5
            right_vel = 1.0
        elif self.inputs[1] < self.inputs[0] and self.inputs[1] < self.inputs[2]:
            # 线在中间，直行
            left_vel = 1.0
            right_vel = 1.0
        elif self.inputs[2] < self.inputs[0] and self.inputs[2] < self.inputs[1]:
            # 线在右边，右转
            left_vel = 1.0
            right_vel = 0.5
        else:
            # 默认：直行
            left_vel = 0.8
            right_vel = 0.8

        return left_vel, right_vel

    def sense_compute_and_actuate(self):
        """
        主控制函数
        """
        if len(self.inputs) > 0 and len(self.inputsPrevious) > 0:
            # 打印调试信息
            self.print_sensor_debug()
            
            # 使用障碍物避障行为
            self.velocity_left, self.velocity_right = self.obstacle_avoidance_behavior()

        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)

    def run_robot(self):
        """
        主循环
        """
        count = 0
        inputs_avg = []
        
        print("\nStarting main loop...")
        print("Waiting for sensor data...\n")
        
        while self.robot.step(self.time_step) != -1:
            # ========== 读取地面传感器 ==========
            self.inputs = []
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()

            # 归一化地面传感器
            min_gs = 0
            max_gs = 1000
            left = max(min_gs, min(max_gs, left))
            center = max(min_gs, min(max_gs, center))
            right = max(min_gs, min(max_gs, right))

            self.inputs.append((left - min_gs) / (max_gs - min_gs))
            self.inputs.append((center - min_gs) / (max_gs - min_gs))
            self.inputs.append((right - min_gs) / (max_gs - min_gs))

            # ========== 读取距离传感器 ==========
            # 按照原始顺序读取ps0, ps1, ps2, ps5, ps6, ps7
            sensor_indices = [0, 1, 2, 5, 6, 7]
            
            for i in range(8):
                self.distance_sensors_raw[i] = self.proximity_sensors[i].getValue()
            
            for i in sensor_indices:
                temp = self.distance_sensors_raw[i]
                
                # 归一化距离传感器
                min_ds = 0
                max_ds = 2400
                temp = max(min_ds, min(max_ds, temp))
                
                normalized_value = (temp - min_ds) / (max_ds - min_ds)
                self.inputs.append(normalized_value)

            # ========== 平滑滤波 ==========
            smooth = 30
            if count == smooth:
                inputs_avg = [sum(x) for x in zip(*inputs_avg)]
                self.inputs = [x / smooth for x in inputs_avg]
                
                # 执行控制
                self.sense_compute_and_actuate()
                
                # 重置
                count = 0
                inputs_avg = []
                self.inputsPrevious = self.inputs
            else:
                inputs_avg.append(self.inputs)
                count = count + 1


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("E-puck Line Following with Obstacle Avoidance")
    print("FIXED VERSION - Enhanced Obstacle Detection")
    print("=" * 60 + "\n")
    
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_robot()
