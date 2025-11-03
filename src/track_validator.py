"""
赛道验证模块 - 检测机器人是否在赛道上

解决的问题：
1. 如何判断机器人是否在赛道上？
2. 机器人偏离赛道时如何计算距离？
3. 如何给予合理的惩罚？

方法：
- 地面传感器检测法：通过地面传感器检测黑线
- 几何检测法：根据赛道形状判断位置
- 混合检测法：结合多种方法

核心思想：
- 在赛道上：正常计算距离，给予奖励
- 偏离赛道：仍计算距离，但给予惩罚，降低适应度
- 完全偏离：大幅降低适应度，鼓励回到赛道
"""
import numpy as np
import math


class TrackValidator:
    """
    赛道验证器基类
    """

    def __init__(self):
        self.on_track_time = 0      # 在赛道上的时间
        self.off_track_time = 0     # 偏离赛道的时间
        self.total_time = 0         # 总时间

        self.on_track_distance = 0.0    # 在赛道上的距离
        self.off_track_distance = 0.0   # 偏离赛道的距离

        self.consecutive_off_track = 0  # 连续偏离次数

    def reset(self):
        """重置验证器"""
        self.on_track_time = 0
        self.off_track_time = 0
        self.total_time = 0
        self.on_track_distance = 0.0
        self.off_track_distance = 0.0
        self.consecutive_off_track = 0

    def is_on_track(self, position, sensor_data):
        """
        判断机器人是否在赛道上

        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据

        Returns:
            bool: 是否在赛道上
        """
        raise NotImplementedError

    def update(self, position, sensor_data, distance_moved):
        """
        更新验证器

        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据
            distance_moved: 本次移动的距离

        Returns:
            dict: 验证结果
        """
        self.total_time += 1

        on_track = self.is_on_track(position, sensor_data)

        if on_track:
            self.on_track_time += 1
            self.on_track_distance += distance_moved
            self.consecutive_off_track = 0
        else:
            self.off_track_time += 1
            self.off_track_distance += distance_moved
            self.consecutive_off_track += 1

        return {
            'on_track': on_track,
            'on_track_ratio': self.get_on_track_ratio(),
            'consecutive_off_track': self.consecutive_off_track
        }

    def get_on_track_ratio(self):
        """获取在赛道上的时间比例"""
        if self.total_time == 0:
            return 1.0
        return self.on_track_time / self.total_time

    def get_track_distance_ratio(self):
        """获取在赛道上的距离比例"""
        total_distance = self.on_track_distance + self.off_track_distance
        if total_distance == 0:
            return 1.0
        return self.on_track_distance / total_distance


class GroundSensorTrackValidator(TrackValidator):
    """
    基于地面传感器的赛道验证器

    原理：
    - 地面传感器检测到黑线 → 在赛道上
    - 所有地面传感器都检测不到黑线 → 偏离赛道

    优点：简单直接，不需要知道赛道形状
    缺点：依赖地面传感器
    """

    def __init__(self, threshold=200):
        """
        初始化

        Args:
            threshold: 地面传感器阈值，高于此值认为检测到黑线
        """
        super().__init__()
        self.threshold = threshold

    def is_on_track(self, position, sensor_data):
        """
        判断是否在赛道上

        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据字典，必须包含'ground_sensors'

        Returns:
            bool: 是否在赛道上
        """
        if 'ground_sensors' not in sensor_data:
            return False

        ground_sensors = sensor_data['ground_sensors']

        # 如果任何一个地面传感器检测到黑线，认为在赛道上
        max_value = max(ground_sensors) if ground_sensors else 0

        return max_value > self.threshold


class GeometricTrackValidator(TrackValidator):
    """
    基于几何的赛道验证器

    原理：
    - 定义赛道的几何形状（如圆环、矩形等）
    - 检查机器人位置是否在赛道区域内

    优点：不依赖传感器，更可靠
    缺点：需要知道赛道的准确形状
    """

    def __init__(self, track_type='circular', **kwargs):
        """
        初始化

        Args:
            track_type: 赛道类型 ('circular', 'rectangular', 'oval')
            **kwargs: 赛道参数
        """
        super().__init__()
        self.track_type = track_type
        self.track_params = kwargs

    def is_on_track(self, position, sensor_data):
        """
        判断是否在赛道上

        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据（此方法不使用）

        Returns:
            bool: 是否在赛道上
        """
        if self.track_type == 'circular':
            return self._is_on_circular_track(position)
        elif self.track_type == 'rectangular':
            return self._is_on_rectangular_track(position)
        elif self.track_type == 'oval':
            return self._is_on_oval_track(position)
        else:
            return False

    def _is_on_circular_track(self, position):
        """
        检查是否在圆形赛道上

        圆形赛道定义：
        - center: 中心点 [x, y]
        - inner_radius: 内半径
        - outer_radius: 外半径

        机器人在内外半径之间即为在赛道上
        """
        center = self.track_params.get('center', [0, 0])
        inner_radius = self.track_params.get('inner_radius', 0.8)
        outer_radius = self.track_params.get('outer_radius', 1.5)

        # 计算到中心的距离
        distance_to_center = np.linalg.norm(
            np.array(position[:2]) - np.array(center)
        )

        # 在内外半径之间
        return inner_radius <= distance_to_center <= outer_radius

    def _is_on_rectangular_track(self, position):
        """
        检查是否在矩形赛道上

        矩形赛道定义：
        - center: 中心点 [x, y]
        - outer_width, outer_height: 外矩形尺寸
        - inner_width, inner_height: 内矩形尺寸
        """
        center = self.track_params.get('center', [0, 0])
        outer_width = self.track_params.get('outer_width', 3.0)
        outer_height = self.track_params.get('outer_height', 2.0)
        inner_width = self.track_params.get('inner_width', 2.0)
        inner_height = self.track_params.get('inner_height', 1.0)

        rel_x = abs(position[0] - center[0])
        rel_y = abs(position[1] - center[1])

        # 在外矩形内
        in_outer = (rel_x <= outer_width / 2 and rel_y <= outer_height / 2)

        # 在内矩形外
        out_inner = (rel_x >= inner_width / 2 or rel_y >= inner_height / 2)

        return in_outer and out_inner

    def _is_on_oval_track(self, position):
        """
        检查是否在椭圆形赛道上
        """
        # 简化：使用圆形逻辑
        return self._is_on_circular_track(position)


class HybridTrackValidator(TrackValidator):
    """
    混合赛道验证器

    结合地面传感器和几何检测

    策略：
    1. 优先使用地面传感器（更准确）
    2. 如果传感器不可用，使用几何检测
    3. 两者结合，提高可靠性
    """

    def __init__(self, track_type='circular', sensor_threshold=200, **track_params):
        """
        初始化

        Args:
            track_type: 赛道类型
            sensor_threshold: 地面传感器阈值
            **track_params: 赛道几何参数
        """
        super().__init__()

        self.sensor_validator = GroundSensorTrackValidator(sensor_threshold)
        self.geometric_validator = GeometricTrackValidator(track_type, **track_params)

        self.use_sensor = True
        self.use_geometric = True

    def is_on_track(self, position, sensor_data):
        """
        判断是否在赛道上

        策略：
        - 如果有地面传感器数据，主要依靠传感器
        - 同时检查几何位置，如果严重偏离则判定为离开赛道

        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据

        Returns:
            bool: 是否在赛道上
        """
        sensor_on_track = False
        geometric_on_track = False

        # 检查地面传感器
        if self.use_sensor and 'ground_sensors' in sensor_data:
            sensor_on_track = self.sensor_validator.is_on_track(position, sensor_data)

        # 检查几何位置
        if self.use_geometric:
            geometric_on_track = self.geometric_validator.is_on_track(position, sensor_data)

        # 组合策略
        if self.use_sensor and self.use_geometric:
            # 两者都要满足（更严格）
            # 或者：至少一个满足（更宽松）
            # 这里使用"至少一个满足"
            return sensor_on_track or geometric_on_track
        elif self.use_sensor:
            return sensor_on_track
        elif self.use_geometric:
            return geometric_on_track
        else:
            return False


class SmartDistanceCalculator:
    """
    智能距离计算器

    根据机器人是否在赛道上，采用不同的距离计算策略

    策略：
    1. 在赛道上：正常计算距离，全额计入
    2. 偏离赛道：仍计算距离，但打折扣
    3. 严重偏离：距离大幅打折，鼓励回到赛道
    """

    def __init__(self, track_validator,
                 on_track_weight=1.0,
                 off_track_weight=0.3,
                 severe_off_track_weight=0.1):
        """
        初始化

        Args:
            track_validator: 赛道验证器
            on_track_weight: 在赛道上的距离权重
            off_track_weight: 偏离赛道的距离权重
            severe_off_track_weight: 严重偏离的距离权重
        """
        self.validator = track_validator

        self.on_track_weight = on_track_weight
        self.off_track_weight = off_track_weight
        self.severe_off_track_weight = severe_off_track_weight

        self.total_distance = 0.0
        self.weighted_distance = 0.0

        self.severe_off_track_threshold = 10  # 连续偏离10次算严重偏离

    def reset(self):
        """重置计算器"""
        self.validator.reset()
        self.total_distance = 0.0
        self.weighted_distance = 0.0

    def update(self, position, sensor_data, distance_moved):
        """
        更新距离计算

        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据
            distance_moved: 本次移动的距离

        Returns:
            dict: 包含各种距离信息
        """
        # 更新验证器
        validation = self.validator.update(position, sensor_data, distance_moved)

        # 累积总距离
        self.total_distance += distance_moved

        # 根据状态计算加权距离
        if validation['on_track']:
            # 在赛道上：全额计入
            weight = self.on_track_weight
        elif validation['consecutive_off_track'] >= self.severe_off_track_threshold:
            # 严重偏离：大幅打折
            weight = self.severe_off_track_weight
        else:
            # 轻微偏离：适度打折
            weight = self.off_track_weight

        weighted_dist = distance_moved * weight
        self.weighted_distance += weighted_dist

        return {
            'total_distance': self.total_distance,
            'weighted_distance': self.weighted_distance,
            'current_weight': weight,
            'on_track': validation['on_track'],
            'on_track_ratio': validation['on_track_ratio']
        }

    def get_effective_distance(self):
        """
        获取有效距离（用于适应度计算）

        Returns:
            float: 加权后的有效距离
        """
        return self.weighted_distance

    def get_stats(self):
        """
        获取统计信息

        Returns:
            dict: 详细统计
        """
        return {
            'total_distance': self.total_distance,
            'weighted_distance': self.weighted_distance,
            'on_track_ratio': self.validator.get_on_track_ratio(),
            'on_track_distance': self.validator.on_track_distance,
            'off_track_distance': self.validator.off_track_distance,
            'efficiency': self.weighted_distance / max(0.001, self.total_distance)
        }


# ==================== 使用示例 ====================

def example_ground_sensor_validation():
    """
    示例1：使用地面传感器验证
    """
    print("示例1: 地面传感器验证")
    print("-" * 60)

    validator = GroundSensorTrackValidator(threshold=200)

    # 模拟传感器数据
    test_cases = [
        {'ground_sensors': [500, 600, 400], 'desc': '在黑线上'},
        {'ground_sensors': [100, 150, 120], 'desc': '偏离黑线'},
        {'ground_sensors': [300, 800, 250], 'desc': '中间传感器在黑线上'},
    ]

    for case in test_cases:
        position = [0, 0, 0]
        on_track = validator.is_on_track(position, case)
        print(f"{case['desc']}: {'✓ 在赛道上' if on_track else '✗ 偏离赛道'}")
        print(f"  传感器值: {case['ground_sensors']}")


def example_geometric_validation():
    """
    示例2：使用几何验证
    """
    print("\n示例2: 几何验证（圆形赛道）")
    print("-" * 60)

    validator = GeometricTrackValidator(
        track_type='circular',
        center=[0, 0],
        inner_radius=0.8,
        outer_radius=1.5
    )

    # 测试不同位置
    test_positions = [
        ([1.0, 0, 0], '在赛道上'),
        ([0.5, 0, 0], '太靠内侧'),
        ([2.0, 0, 0], '太靠外侧'),
        ([1.2, 0, 0], '在赛道上'),
    ]

    for position, desc in test_positions:
        on_track = validator.is_on_track(position, {})
        distance = np.linalg.norm(position[:2])
        print(f"{desc}: {'✓' if on_track else '✗'} (距中心 {distance:.2f}m)")


def example_smart_distance_calculation():
    """
    示例3：智能距离计算
    """
    print("\n示例3: 智能距离计算")
    print("-" * 60)

    # 创建混合验证器
    validator = HybridTrackValidator(
        track_type='circular',
        center=[0, 0],
        inner_radius=0.8,
        outer_radius=1.5,
        sensor_threshold=200
    )

    # 创建智能距离计算器
    calculator = SmartDistanceCalculator(
        validator,
        on_track_weight=1.0,      # 在赛道上：100%
        off_track_weight=0.3,     # 偏离赛道：30%
        severe_off_track_weight=0.1  # 严重偏离：10%
    )

    # 模拟机器人运动
    scenarios = [
        # (位置, 传感器数据, 移动距离, 描述)
        ([1.0, 0, 0], {'ground_sensors': [500, 600, 400]}, 0.1, '在赛道上'),
        ([1.1, 0.1, 0], {'ground_sensors': [500, 550, 450]}, 0.1, '在赛道上'),
        ([1.8, 0.2, 0], {'ground_sensors': [100, 120, 110]}, 0.1, '偏离赛道'),
        ([2.0, 0.3, 0], {'ground_sensors': [80, 90, 85]}, 0.1, '偏离赛道'),
        ([2.2, 0.4, 0], {'ground_sensors': [70, 75, 70]}, 0.1, '严重偏离'),
    ]

    print("\n机器人运动轨迹:")
    for position, sensor_data, distance, desc in scenarios:
        result = calculator.update(position, sensor_data, distance)

        print(f"\n{desc}:")
        print(f"  位置: ({position[0]:.2f}, {position[1]:.2f})")
        print(f"  实际移动: {distance:.3f}m")
        print(f"  有效距离: {distance * result['current_weight']:.3f}m (权重: {result['current_weight']:.1%})")
        print(f"  在赛道上: {'是' if result['on_track'] else '否'}")

    # 最终统计
    stats = calculator.get_stats()
    print("\n" + "=" * 60)
    print("最终统计:")
    print(f"  总距离: {stats['total_distance']:.3f}m")
    print(f"  有效距离: {stats['weighted_distance']:.3f}m")
    print(f"  在赛道比例: {stats['on_track_ratio']:.1%}")
    print(f"  效率: {stats['efficiency']:.1%}")


def example_integration():
    """
    示例4：与适应度评估器集成
    """
    print("\n示例4: 与适应度评估器集成")
    print("-" * 60)

    # 这展示了如何在实际训练中使用

    # 1. 创建赛道验证器
    validator = HybridTrackValidator(
        track_type='circular',
        center=[0, 0],
        inner_radius=0.8,
        outer_radius=1.5,
        sensor_threshold=200
    )

    # 2. 创建智能距离计算器
    distance_calc = SmartDistanceCalculator(validator)

    # 3. 在训练循环中使用
    print("\n在训练循环中:")
    print("```python")
    print("while training:")
    print("    # 获取机器人状态")
    print("    position = get_robot_position()")
    print("    sensor_data = read_sensors()")
    print("    distance_moved = calculate_distance_moved()")
    print("    ")
    print("    # 更新智能距离计算器")
    print("    result = distance_calc.update(position, sensor_data, distance_moved)")
    print("    ")
    print("    # 使用有效距离计算适应度")
    print("    effective_distance = distance_calc.get_effective_distance()")
    print("    fitness = calculate_fitness(effective_distance, ...)")
    print("    ")
    print("    # 获取统计信息")
    print("    stats = distance_calc.get_stats()")
    print("    on_track_ratio = stats['on_track_ratio']")
    print("```")


if __name__ == '__main__':
    print("=" * 60)
    print("赛道验证和智能距离计算示例")
    print("=" * 60)

    example_ground_sensor_validation()
    example_geometric_validation()
    example_smart_distance_calculation()
    example_integration()

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
"""
赛道验证模块 - 检测机器人是否在赛道上

解决的问题：
1. 如何判断机器人是否在赛道上？
2. 机器人偏离赛道时如何计算距离？
3. 如何给予合理的惩罚？

方法：
- 地面传感器检测法：通过地面传感器检测黑线
- 几何检测法：根据赛道形状判断位置
- 混合检测法：结合多种方法

核心思想：
- 在赛道上：正常计算距离，给予奖励
- 偏离赛道：仍计算距离，但给予惩罚，降低适应度
- 完全偏离：大幅降低适应度，鼓励回到赛道
"""
import numpy as np
import math


class TrackValidator:
    """
    赛道验证器基类
    """
    
    def __init__(self):
        self.on_track_time = 0      # 在赛道上的时间
        self.off_track_time = 0     # 偏离赛道的时间
        self.total_time = 0         # 总时间
        
        self.on_track_distance = 0.0    # 在赛道上的距离
        self.off_track_distance = 0.0   # 偏离赛道的距离
        
        self.consecutive_off_track = 0  # 连续偏离次数
    
    def reset(self):
        """重置验证器"""
        self.on_track_time = 0
        self.off_track_time = 0
        self.total_time = 0
        self.on_track_distance = 0.0
        self.off_track_distance = 0.0
        self.consecutive_off_track = 0
    
    def is_on_track(self, position, sensor_data):
        """
        判断机器人是否在赛道上
        
        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据
            
        Returns:
            bool: 是否在赛道上
        """
        raise NotImplementedError
    
    def update(self, position, sensor_data, distance_moved):
        """
        更新验证器
        
        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据
            distance_moved: 本次移动的距离
            
        Returns:
            dict: 验证结果
        """
        self.total_time += 1
        
        on_track = self.is_on_track(position, sensor_data)
        
        if on_track:
            self.on_track_time += 1
            self.on_track_distance += distance_moved
            self.consecutive_off_track = 0
        else:
            self.off_track_time += 1
            self.off_track_distance += distance_moved
            self.consecutive_off_track += 1
        
        return {
            'on_track': on_track,
            'on_track_ratio': self.get_on_track_ratio(),
            'consecutive_off_track': self.consecutive_off_track
        }
    
    def get_on_track_ratio(self):
        """获取在赛道上的时间比例"""
        if self.total_time == 0:
            return 1.0
        return self.on_track_time / self.total_time
    
    def get_track_distance_ratio(self):
        """获取在赛道上的距离比例"""
        total_distance = self.on_track_distance + self.off_track_distance
        if total_distance == 0:
            return 1.0
        return self.on_track_distance / total_distance


class GroundSensorTrackValidator(TrackValidator):
    """
    基于地面传感器的赛道验证器
    
    原理：
    - 地面传感器检测到黑线 → 在赛道上
    - 所有地面传感器都检测不到黑线 → 偏离赛道
    
    优点：简单直接，不需要知道赛道形状
    缺点：依赖地面传感器
    """
    
    def __init__(self, threshold=200):
        """
        初始化
        
        Args:
            threshold: 地面传感器阈值，高于此值认为检测到黑线
        """
        super().__init__()
        self.threshold = threshold
    
    def is_on_track(self, position, sensor_data):
        """
        判断是否在赛道上
        
        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据字典，必须包含'ground_sensors'
            
        Returns:
            bool: 是否在赛道上
        """
        if 'ground_sensors' not in sensor_data:
            return False
        
        ground_sensors = sensor_data['ground_sensors']
        
        # 如果任何一个地面传感器检测到黑线，认为在赛道上
        max_value = max(ground_sensors) if ground_sensors else 0
        
        return max_value > self.threshold


class GeometricTrackValidator(TrackValidator):
    """
    基于几何的赛道验证器
    
    原理：
    - 定义赛道的几何形状（如圆环、矩形等）
    - 检查机器人位置是否在赛道区域内
    
    优点：不依赖传感器，更可靠
    缺点：需要知道赛道的准确形状
    """
    
    def __init__(self, track_type='circular', **kwargs):
        """
        初始化
        
        Args:
            track_type: 赛道类型 ('circular', 'rectangular', 'oval')
            **kwargs: 赛道参数
        """
        super().__init__()
        self.track_type = track_type
        self.track_params = kwargs
    
    def is_on_track(self, position, sensor_data):
        """
        判断是否在赛道上
        
        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据（此方法不使用）
            
        Returns:
            bool: 是否在赛道上
        """
        if self.track_type == 'circular':
            return self._is_on_circular_track(position)
        elif self.track_type == 'rectangular':
            return self._is_on_rectangular_track(position)
        elif self.track_type == 'oval':
            return self._is_on_oval_track(position)
        else:
            return False
    
    def _is_on_circular_track(self, position):
        """
        检查是否在圆形赛道上
        
        圆形赛道定义：
        - center: 中心点 [x, y]
        - inner_radius: 内半径
        - outer_radius: 外半径
        
        机器人在内外半径之间即为在赛道上
        """
        center = self.track_params.get('center', [0, 0])
        inner_radius = self.track_params.get('inner_radius', 0.8)
        outer_radius = self.track_params.get('outer_radius', 1.5)
        
        # 计算到中心的距离
        distance_to_center = np.linalg.norm(
            np.array(position[:2]) - np.array(center)
        )
        
        # 在内外半径之间
        return inner_radius <= distance_to_center <= outer_radius
    
    def _is_on_rectangular_track(self, position):
        """
        检查是否在矩形赛道上
        
        矩形赛道定义：
        - center: 中心点 [x, y]
        - outer_width, outer_height: 外矩形尺寸
        - inner_width, inner_height: 内矩形尺寸
        """
        center = self.track_params.get('center', [0, 0])
        outer_width = self.track_params.get('outer_width', 3.0)
        outer_height = self.track_params.get('outer_height', 2.0)
        inner_width = self.track_params.get('inner_width', 2.0)
        inner_height = self.track_params.get('inner_height', 1.0)
        
        rel_x = abs(position[0] - center[0])
        rel_y = abs(position[1] - center[1])
        
        # 在外矩形内
        in_outer = (rel_x <= outer_width / 2 and rel_y <= outer_height / 2)
        
        # 在内矩形外
        out_inner = (rel_x >= inner_width / 2 or rel_y >= inner_height / 2)
        
        return in_outer and out_inner
    
    def _is_on_oval_track(self, position):
        """
        检查是否在椭圆形赛道上
        """
        # 简化：使用圆形逻辑
        return self._is_on_circular_track(position)


class HybridTrackValidator(TrackValidator):
    """
    混合赛道验证器
    
    结合地面传感器和几何检测
    
    策略：
    1. 优先使用地面传感器（更准确）
    2. 如果传感器不可用，使用几何检测
    3. 两者结合，提高可靠性
    """
    
    def __init__(self, track_type='circular', sensor_threshold=200, **track_params):
        """
        初始化
        
        Args:
            track_type: 赛道类型
            sensor_threshold: 地面传感器阈值
            **track_params: 赛道几何参数
        """
        super().__init__()
        
        self.sensor_validator = GroundSensorTrackValidator(sensor_threshold)
        self.geometric_validator = GeometricTrackValidator(track_type, **track_params)
        
        self.use_sensor = True
        self.use_geometric = True
    
    def is_on_track(self, position, sensor_data):
        """
        判断是否在赛道上
        
        策略：
        - 如果有地面传感器数据，主要依靠传感器
        - 同时检查几何位置，如果严重偏离则判定为离开赛道
        
        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据
            
        Returns:
            bool: 是否在赛道上
        """
        sensor_on_track = False
        geometric_on_track = False
        
        # 检查地面传感器
        if self.use_sensor and 'ground_sensors' in sensor_data:
            sensor_on_track = self.sensor_validator.is_on_track(position, sensor_data)
        
        # 检查几何位置
        if self.use_geometric:
            geometric_on_track = self.geometric_validator.is_on_track(position, sensor_data)
        
        # 组合策略
        if self.use_sensor and self.use_geometric:
            # 两者都要满足（更严格）
            # 或者：至少一个满足（更宽松）
            # 这里使用"至少一个满足"
            return sensor_on_track or geometric_on_track
        elif self.use_sensor:
            return sensor_on_track
        elif self.use_geometric:
            return geometric_on_track
        else:
            return False


class SmartDistanceCalculator:
    """
    智能距离计算器
    
    根据机器人是否在赛道上，采用不同的距离计算策略
    
    策略：
    1. 在赛道上：正常计算距离，全额计入
    2. 偏离赛道：仍计算距离，但打折扣
    3. 严重偏离：距离大幅打折，鼓励回到赛道
    """
    
    def __init__(self, track_validator, 
                 on_track_weight=1.0,
                 off_track_weight=0.3,
                 severe_off_track_weight=0.1):
        """
        初始化
        
        Args:
            track_validator: 赛道验证器
            on_track_weight: 在赛道上的距离权重
            off_track_weight: 偏离赛道的距离权重
            severe_off_track_weight: 严重偏离的距离权重
        """
        self.validator = track_validator
        
        self.on_track_weight = on_track_weight
        self.off_track_weight = off_track_weight
        self.severe_off_track_weight = severe_off_track_weight
        
        self.total_distance = 0.0
        self.weighted_distance = 0.0
        
        self.severe_off_track_threshold = 10  # 连续偏离10次算严重偏离
    
    def reset(self):
        """重置计算器"""
        self.validator.reset()
        self.total_distance = 0.0
        self.weighted_distance = 0.0
    
    def update(self, position, sensor_data, distance_moved):
        """
        更新距离计算
        
        Args:
            position: 机器人位置 [x, y, z]
            sensor_data: 传感器数据
            distance_moved: 本次移动的距离
            
        Returns:
            dict: 包含各种距离信息
        """
        # 更新验证器
        validation = self.validator.update(position, sensor_data, distance_moved)
        
        # 累积总距离
        self.total_distance += distance_moved
        
        # 根据状态计算加权距离
        if validation['on_track']:
            # 在赛道上：全额计入
            weight = self.on_track_weight
        elif validation['consecutive_off_track'] >= self.severe_off_track_threshold:
            # 严重偏离：大幅打折
            weight = self.severe_off_track_weight
        else:
            # 轻微偏离：适度打折
            weight = self.off_track_weight
        
        weighted_dist = distance_moved * weight
        self.weighted_distance += weighted_dist
        
        return {
            'total_distance': self.total_distance,
            'weighted_distance': self.weighted_distance,
            'current_weight': weight,
            'on_track': validation['on_track'],
            'on_track_ratio': validation['on_track_ratio']
        }
    
    def get_effective_distance(self):
        """
        获取有效距离（用于适应度计算）
        
        Returns:
            float: 加权后的有效距离
        """
        return self.weighted_distance
    
    def get_stats(self):
        """
        获取统计信息
        
        Returns:
            dict: 详细统计
        """
        return {
            'total_distance': self.total_distance,
            'weighted_distance': self.weighted_distance,
            'on_track_ratio': self.validator.get_on_track_ratio(),
            'on_track_distance': self.validator.on_track_distance,
            'off_track_distance': self.validator.off_track_distance,
            'efficiency': self.weighted_distance / max(0.001, self.total_distance)
        }


# ==================== 使用示例 ====================

def example_ground_sensor_validation():
    """
    示例1：使用地面传感器验证
    """
    print("示例1: 地面传感器验证")
    print("-" * 60)
    
    validator = GroundSensorTrackValidator(threshold=200)
    
    # 模拟传感器数据
    test_cases = [
        {'ground_sensors': [500, 600, 400], 'desc': '在黑线上'},
        {'ground_sensors': [100, 150, 120], 'desc': '偏离黑线'},
        {'ground_sensors': [300, 800, 250], 'desc': '中间传感器在黑线上'},
    ]
    
    for case in test_cases:
        position = [0, 0, 0]
        on_track = validator.is_on_track(position, case)
        print(f"{case['desc']}: {'✓ 在赛道上' if on_track else '✗ 偏离赛道'}")
        print(f"  传感器值: {case['ground_sensors']}")


def example_geometric_validation():
    """
    示例2：使用几何验证
    """
    print("\n示例2: 几何验证（圆形赛道）")
    print("-" * 60)
    
    validator = GeometricTrackValidator(
        track_type='circular',
        center=[0, 0],
        inner_radius=0.8,
        outer_radius=1.5
    )
    
    # 测试不同位置
    test_positions = [
        ([1.0, 0, 0], '在赛道上'),
        ([0.5, 0, 0], '太靠内侧'),
        ([2.0, 0, 0], '太靠外侧'),
        ([1.2, 0, 0], '在赛道上'),
    ]
    
    for position, desc in test_positions:
        on_track = validator.is_on_track(position, {})
        distance = np.linalg.norm(position[:2])
        print(f"{desc}: {'✓' if on_track else '✗'} (距中心 {distance:.2f}m)")


def example_smart_distance_calculation():
    """
    示例3：智能距离计算
    """
    print("\n示例3: 智能距离计算")
    print("-" * 60)
    
    # 创建混合验证器
    validator = HybridTrackValidator(
        track_type='circular',
        center=[0, 0],
        inner_radius=0.8,
        outer_radius=1.5,
        sensor_threshold=200
    )
    
    # 创建智能距离计算器
    calculator = SmartDistanceCalculator(
        validator,
        on_track_weight=1.0,      # 在赛道上：100%
        off_track_weight=0.3,     # 偏离赛道：30%
        severe_off_track_weight=0.1  # 严重偏离：10%
    )
    
    # 模拟机器人运动
    scenarios = [
        # (位置, 传感器数据, 移动距离, 描述)
        ([1.0, 0, 0], {'ground_sensors': [500, 600, 400]}, 0.1, '在赛道上'),
        ([1.1, 0.1, 0], {'ground_sensors': [500, 550, 450]}, 0.1, '在赛道上'),
        ([1.8, 0.2, 0], {'ground_sensors': [100, 120, 110]}, 0.1, '偏离赛道'),
        ([2.0, 0.3, 0], {'ground_sensors': [80, 90, 85]}, 0.1, '偏离赛道'),
        ([2.2, 0.4, 0], {'ground_sensors': [70, 75, 70]}, 0.1, '严重偏离'),
    ]
    
    print("\n机器人运动轨迹:")
    for position, sensor_data, distance, desc in scenarios:
        result = calculator.update(position, sensor_data, distance)
        
        print(f"\n{desc}:")
        print(f"  位置: ({position[0]:.2f}, {position[1]:.2f})")
        print(f"  实际移动: {distance:.3f}m")
        print(f"  有效距离: {distance * result['current_weight']:.3f}m (权重: {result['current_weight']:.1%})")
        print(f"  在赛道上: {'是' if result['on_track'] else '否'}")
    
    # 最终统计
    stats = calculator.get_stats()
    print("\n" + "=" * 60)
    print("最终统计:")
    print(f"  总距离: {stats['total_distance']:.3f}m")
    print(f"  有效距离: {stats['weighted_distance']:.3f}m")
    print(f"  在赛道比例: {stats['on_track_ratio']:.1%}")
    print(f"  效率: {stats['efficiency']:.1%}")


def example_integration():
    """
    示例4：与适应度评估器集成
    """
    print("\n示例4: 与适应度评估器集成")
    print("-" * 60)
    
    # 这展示了如何在实际训练中使用
    
    # 1. 创建赛道验证器
    validator = HybridTrackValidator(
        track_type='circular',
        center=[0, 0],
        inner_radius=0.8,
        outer_radius=1.5,
        sensor_threshold=200
    )
    
    # 2. 创建智能距离计算器
    distance_calc = SmartDistanceCalculator(validator)
    
    # 3. 在训练循环中使用
    print("\n在训练循环中:")
    print("```python")
    print("while training:")
    print("    # 获取机器人状态")
    print("    position = get_robot_position()")
    print("    sensor_data = read_sensors()")
    print("    distance_moved = calculate_distance_moved()")
    print("    ")
    print("    # 更新智能距离计算器")
    print("    result = distance_calc.update(position, sensor_data, distance_moved)")
    print("    ")
    print("    # 使用有效距离计算适应度")
    print("    effective_distance = distance_calc.get_effective_distance()")
    print("    fitness = calculate_fitness(effective_distance, ...)")
    print("    ")
    print("    # 获取统计信息")
    print("    stats = distance_calc.get_stats()")
    print("    on_track_ratio = stats['on_track_ratio']")
    print("```")


if __name__ == '__main__':
    print("=" * 60)
    print("赛道验证和智能距离计算示例")
    print("=" * 60)
    
    example_ground_sensor_validation()
    example_geometric_validation()
    example_smart_distance_calculation()
    example_integration()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
