"""
E-puck传感器测试脚本

这个脚本用于测试所有传感器是否正常工作
包括：
1. 8个距离传感器 (ps0-ps7)
2. 3个地面传感器 (gs0-gs2)
3. 实时显示传感器值
4. 可视化传感器状态

使用方法：
1. 将此脚本设置为E-puck的控制器
2. 运行Webots
3. 观察控制台输出
4. 手动在机器人前方放置障碍物，观察传感器值变化
"""
from controller import Robot
import time


class SensorTester:
    def __init__(self, robot):
        self.robot = robot
        self.time_step = 32

        # 初始化电机（保持静止）
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf')
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # 初始化距离传感器
        self.distance_sensors = []
        self.distance_sensor_names = []

        print("\n" + "=" * 70)
        print("初始化距离传感器...")
        print("=" * 70)

        for i in range(8):
            sensor_name = 'ps' + str(i)
        try:
            sensor = self.robot.getDevice(sensor_name)
            if sensor is None:
                print(f"❌ 传感器 {sensor_name} 未找到！")
                self.distance_sensors.append(None)
            else:
                sensor.enable(self.time_step)
                self.distance_sensors.append(sensor)
                self.distance_sensor_names.append(sensor_name)
                print(f"✅ 传感器 {sensor_name} 初始化成功")
        except Exception as e:
            print(f"❌ 传感器 {sensor_name} 初始化失败: {e}")
            self.distance_sensors.append(None)

        # 初始化地面传感器
        print("\n" + "=" * 70)
        print("初始化地面传感器...")
        print("=" * 70)

        self.ground_sensors = []
        self.ground_sensor_names = ['gs0', 'gs1', 'gs2']

        for name in self.ground_sensor_names:
            try:
                sensor = self.robot.getDevice(name)
                if sensor is None:
                    print(f"❌ 地面传感器 {name} 未找到！")
                    self.ground_sensors.append(None)
                else:
                    sensor.enable(self.time_step)
                    self.ground_sensors.append(sensor)
                    print(f"✅ 地面传感器 {name} 初始化成功")
            except Exception as e:
                print(f"❌ 地面传感器 {name} 初始化失败: {e}")
                self.ground_sensors.append(None)

        print("\n" + "=" * 70)
        print("传感器初始化完成！")
        print("=" * 70)

        # 传感器布局说明
        self.print_sensor_layout()

        # 计数器
        self.step_count = 0

    def print_sensor_layout(self):
        """打印传感器布局说明"""
        print("\n" + "=" * 70)
        print("E-puck传感器布局")
        print("=" * 70)
        print("""
距离传感器布局（俯视图）：

            前方
         ps7    ps6
           \\  //
            \\/
    ps5 ---- 🤖 ---- ps2
            /\\
           //  \\
         ps0    ps1
            后方

        ps3    ps4

说明：
- ps0, ps1: 前右方（主要检测前方右侧障碍物）
- ps2: 右侧（检测右侧障碍物）
- ps3, ps4: 后方（不常用）
- ps5: 左侧（检测左侧障碍物）
- ps6, ps7: 前左方（主要检测前方左侧障碍物）

地面传感器布局：
- gs0: 左侧地面传感器
- gs1: 中间地面传感器
- gs2: 右侧地面传感器
        """)
        print("=" * 70)

    def read_distance_sensors(self):
        """读取所有距离传感器"""
        values = []
        for i, sensor in enumerate(self.distance_sensors):
            if sensor is not None:
                try:
                    value = sensor.getValue()
                    values.append(value)
                except:
                    values.append(-1)
            else:
                values.append(-1)
        return values

    def read_ground_sensors(self):
        """读取所有地面传感器"""
        values = []
        for sensor in self.ground_sensors:
            if sensor is not None:
                try:
                    value = sensor.getValue()
                    values.append(value)
                except:
                    values.append(-1)
            else:
                values.append(-1)
        return values

    def print_sensor_bar(self, value, max_value, width=30):
        """打印传感器值的条形图"""
        if value < 0:
            return "❌ 传感器错误"

        normalized = min(1.0, value / max_value)
        filled = int(normalized * width)
        bar = "█" * filled + "░" * (width - filled)

        # 根据值的大小选择颜色指示
        if normalized > 0.5:
            indicator = "🔴"  # 高值（检测到障碍物）
        elif normalized > 0.2:
            indicator = "🟡"  # 中值
        else:
            indicator = "🟢"  # 低值（无障碍物）

        return f"{indicator} {bar} {value:7.1f} ({normalized * 100:5.1f}%)"

    def print_sensor_status(self, distance_values, ground_values):
        """打印所有传感器状态"""
        print("\n" + "=" * 70)
        print(f"步数: {self.step_count}")
        print("=" * 70)

        # 打印距离传感器
        print("\n【距离传感器】(范围: 0-2400, 值越大=障碍物越近)")
        print("-" * 70)

        for i in range(8):
            if i < len(distance_values):
                value = distance_values[i]
                bar = self.print_sensor_bar(value, 2400)

                # 添加位置说明
                if i == 0:
                    position = "前右"
                elif i == 1:
                    position = "前右"
                elif i == 2:
                    position = "右侧"
                elif i == 3:
                    position = "后右"
                elif i == 4:
                    position = "后左"
                elif i == 5:
                    position = "左侧"
                elif i == 6:
                    position = "前左"
                elif i == 7:
                    position = "前左"
                else:
                    position = "未知"

                print(f"ps{i} ({position:4s}): {bar}")

        # 打印地面传感器
        print("\n【地面传感器】(范围: 0-1000, 值越大=越亮/白色, 值越小=越暗/黑线)")
        print("-" * 70)

        ground_names = ['左侧', '中间', '右侧']
        for i in range(3):
            if i < len(ground_values):
                value = ground_values[i]
                bar = self.print_sensor_bar(value, 1000)
                print(f"gs{i} ({ground_names[i]}): {bar}")

        # 打印前方障碍物检测
        print("\n【前方障碍物检测】")
        print("-" * 70)

        if len(distance_values) >= 8:
            front_sensors = [
                distance_values[0],  # ps0
                distance_values[1],  # ps1
                distance_values[6],  # ps6
                distance_values[7],  # ps7
            ]

            max_front = max(front_sensors)

            if max_front > 500:
                print(f"🔴 检测到前方障碍物！最大值: {max_front:.1f}")
                if distance_values[6] > 500 or distance_values[7] > 500:
                    print(f"   位置: 前左方 (ps6={distance_values[6]:.1f}, ps7={distance_values[7]:.1f})")
                if distance_values[0] > 500 or distance_values[1] > 500:
                    print(f"   位置: 前右方 (ps0={distance_values[0]:.1f}, ps1={distance_values[1]:.1f})")
            elif max_front > 200:
                print(f"🟡 前方可能有障碍物 (最大值: {max_front:.1f})")
            else:
                print(f"🟢 前方无障碍物 (最大值: {max_front:.1f})")

        # 打印黑线检测
        print("\n【黑线检测】")
        print("-" * 70)

        if len(ground_values) >= 3:
            min_ground = min(ground_values)

            if min_ground < 300:
                print(f"🔴 检测到黑线！最小值: {min_ground:.1f}")
                if ground_values[0] < 300:
                    print(f"   位置: 左侧 (gs0={ground_values[0]:.1f})")
                if ground_values[1] < 300:
                    print(f"   位置: 中间 (gs1={ground_values[1]:.1f})")
                if ground_values[2] < 300:
                    print(f"   位置: 右侧 (gs2={ground_values[2]:.1f})")
            else:
                print(f"🟢 未检测到黑线 (最小值: {min_ground:.1f})")

        print("=" * 70)

    def run_test(self):
        """运行传感器测试"""
        print("\n" + "=" * 70)
        print("开始传感器测试")
        print("=" * 70)
        print("\n提示：")
        print("1. 机器人将保持静止")
        print("2. 请在机器人前方放置障碍物，观察传感器值变化")
        print("3. 将机器人放在黑线上，观察地面传感器值变化")
        print("4. 按 Ctrl+C 停止测试")
        print("\n等待传感器数据...\n")

        # 等待几步让传感器初始化
        for _ in range(5):
            self.robot.step(self.time_step)

        try:
            while self.robot.step(self.time_step) != -1:
                self.step_count += 1

                # 读取传感器
                distance_values = self.read_distance_sensors()
                ground_values = self.read_ground_sensors()

                # 每10步打印一次（避免刷屏）
                if self.step_count % 10 == 0:
                    self.print_sensor_status(distance_values, ground_values)

                    # 提示用户
                    if self.step_count == 10:
                        print("\n💡 提示: 现在可以在机器人前方放置障碍物测试！")

        except KeyboardInterrupt:
            print("\n\n测试被用户中断")
            print("=" * 70)
            print("测试结束")
            print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("E-puck传感器测试程序")
    print("=" * 70)

    robot = Robot()
    tester = SensorTester(robot)
    tester.run_test()


if __name__ == "__main__":
    main()
