"""
并行训练器 - 同时评估多个机器人

这个版本使用Webots的Supervisor功能，可以在一个世界文件中
同时运行多个机器人，大大加快训练速度！

核心思想：
1. 使用Supervisor控制器管理多个机器人
2. 每个机器人评估不同的神经网络
3. 所有机器人同时运行，并行评估
4. 评估完成后，重置所有机器人，开始下一批

优势：
- 训练速度提升N倍（N=机器人数量）
- 充分利用计算资源
- 更快看到训练结果
"""
import sys
import time
import numpy as np
from controller import Supervisor
from genetic_algorithm import GeneticAlgorithm
from neural_network import NeuralNetwork
from fitness_evaluator import FitnessEvaluator
from model_utils import save_model, save_checkpoint, save_training_log
from config import GA_CONFIG, ROBOT_CONFIG


class ParallelRobotController:
    """
    单个机器人的控制器（用于并行评估）
    """
    
    def __init__(self, robot_node, robot_id, supervisor):
        """
        初始化机器人控制器
        
        Args:
            robot_node: Webots机器人节点
            robot_id: 机器人ID
            supervisor: Supervisor对象
        """
        self.robot_node = robot_node
        self.robot_id = robot_id
        self.supervisor = supervisor
        
        # 获取机器人设备
        self.left_motor = robot_node.getDevice('left wheel motor')
        self.right_motor = robot_node.getDevice('right wheel motor')
        
        # 设置电机为速度控制模式
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # 获取传感器
        self.distance_sensors = []
        for name in ROBOT_CONFIG['distance_sensor_names']:
            sensor = robot_node.getDevice(name)
            if sensor:
                sensor.enable(int(supervisor.getBasicTimeStep()))
                self.distance_sensors.append(sensor)
        
        self.ground_sensors = []
        for name in ROBOT_CONFIG['ground_sensor_names']:
            sensor = robot_node.getDevice(name)
            if sensor:
                sensor.enable(int(supervisor.getBasicTimeStep()))
                self.ground_sensors.append(sensor)
        
        # 神经网络
        self.neural_network = None
        
        # 适应度评估器
        self.fitness_evaluator = FitnessEvaluator()
        
        # 起始位置
        self.start_position = None
        
        # 最大速度
        self.max_speed = ROBOT_CONFIG['max_speed']
    
    def set_neural_network(self, neural_network):
        """设置神经网络"""
        self.neural_network = neural_network
    
    def reset(self):
        """重置机器人状态"""
        self.fitness_evaluator.reset()
        self.start_position = None
        self.set_motor_speeds(0.0, 0.0)
    
    def get_position(self):
        """获取机器人位置"""
        translation = self.robot_node.getField('translation')
        if translation:
            return translation.getSFVec3f()
        return None
    
    def set_position(self, x, y, z=0.0):
        """设置机器人位置"""
        translation = self.robot_node.getField('translation')
        if translation:
            translation.setSFVec3f([x, y, z])
    
    def set_rotation(self, angle):
        """设置机器人旋转角度"""
        rotation = self.robot_node.getField('rotation')
        if rotation:
            rotation.setSFRotation([0, 0, 1, angle])
    
    def read_sensors(self):
        """读取传感器数据"""
        distance_values = [s.getValue() for s in self.distance_sensors]
        ground_values = [s.getValue() for s in self.ground_sensors]
        position = self.get_position()
        
        return {
            'distance_sensors': distance_values,
            'ground_sensors': ground_values,
            'position': position
        }
    
    def normalize_sensors(self, sensor_data):
        """归一化传感器数据"""
        distance_normalized = np.array(sensor_data['distance_sensors']) / 4096.0
        ground_normalized = np.array(sensor_data['ground_sensors']) / 1000.0
        all_sensors = np.concatenate([distance_normalized, ground_normalized])
        return all_sensors
    
    def set_motor_speeds(self, left_speed, right_speed):
        """设置电机速度"""
        left_velocity = left_speed * self.max_speed
        right_velocity = right_speed * self.max_speed
        self.left_motor.setVelocity(left_velocity)
        self.right_motor.setVelocity(right_velocity)
    
    def step(self):
        """执行一个控制步骤"""
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
    
    def get_fitness(self):
        """获取适应度分数"""
        return self.fitness_evaluator.calculate_fitness()
    
    def get_stats(self):
        """获取统计信息"""
        return self.fitness_evaluator.get_stats()


class ParallelTrainer:
    """
    并行训练器
    
    使用Supervisor同时管理多个机器人进行训练
    """
    
    def __init__(self, num_robots=None):
        """
        初始化并行训练器
        
        Args:
            num_robots: 机器人数量，如果为None则使用种群大小
        """
        print("=" * 70)
        print("并行遗传算法训练器 - E-puck循迹避障")
        print("=" * 70)
        
        # 初始化Supervisor
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # 初始化遗传算法
        self.ga = GeneticAlgorithm(GA_CONFIG)
        
        # 机器人数量
        if num_robots is None:
            num_robots = min(self.ga.population_size, 30)  # 最多30个机器人
        self.num_robots = num_robots
        
        # 训练参数
        self.generations = GA_CONFIG['generations']
        self.simulation_time = GA_CONFIG['simulation_time']
        
        # 初始化机器人控制器
        self.robot_controllers = []
        self._init_robots()
        
        # 训练日志
        self.training_log = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_individual_stats': []
        }
        
        print(f"\n配置:")
        print(f"  种群大小: {self.ga.population_size}")
        print(f"  并行机器人数: {self.num_robots}")
        print(f"  进化代数: {self.generations}")
        print(f"  每次模拟时间: {self.simulation_time}秒")
        print(f"  加速比: {self.num_robots}x")
        print()
    
    def _init_robots(self):
        """初始化所有机器人"""
        print(f"初始化{self.num_robots}个机器人...")
        
        # 获取所有E-puck机器人节点
        root = self.supervisor.getRoot()
        children_field = root.getField('children')
        
        robot_nodes = []
        for i in range(children_field.getCount()):
            node = children_field.getMFNode(i)
            type_name = node.getTypeName()
            
            # 查找E-puck机器人
            if 'E-puck' in type_name or 'Robot' in type_name:
                robot_nodes.append(node)
                if len(robot_nodes) >= self.num_robots:
                    break
        
        if len(robot_nodes) < self.num_robots:
            print(f"警告: 只找到{len(robot_nodes)}个机器人，少于请求的{self.num_robots}个")
            self.num_robots = len(robot_nodes)
        
        # 创建机器人控制器
        for i, robot_node in enumerate(robot_nodes):
            controller = ParallelRobotController(robot_node, i, self.supervisor)
            self.robot_controllers.append(controller)
            print(f"  机器人{i+1}初始化完成")
        
        print(f"成功初始化{len(self.robot_controllers)}个机器人")
    
    def reset_robots_positions(self, spacing=0.3):
        """
        重置所有机器人位置
        
        将机器人排列在起始线上
        
        Args:
            spacing: 机器人之间的间距（米）
        """
        num_robots = len(self.robot_controllers)
        
        # 计算排列位置（一字排开）
        for i, controller in enumerate(self.robot_controllers):
            # 计算位置（以原点为中心，向两侧排列）
            offset = (i - num_robots / 2.0) * spacing
            x = offset
            y = 0.0
            z = 0.0
            
            controller.set_position(x, y, z)
            controller.set_rotation(0.0)  # 朝向前方
            controller.reset()
    
    def evaluate_population_parallel(self):
        """
        并行评估种群
        
        这是核心优化！同时评估多个个体
        
        Returns:
            list: 每个个体的适应度分数
        """
        print(f"\n并行评估第{self.ga.generation}代种群...")
        
        population = self.ga.get_population()
        fitness_scores = [0.0] * len(population)
        
        # 计算需要多少批次
        num_batches = (len(population) + self.num_robots - 1) // self.num_robots
        
        print(f"  总个体数: {len(population)}")
        print(f"  并行数: {self.num_robots}")
        print(f"  批次数: {num_batches}")
        
        # 分批评估
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.num_robots
            end_idx = min(start_idx + self.num_robots, len(population))
            batch_size = end_idx - start_idx
            
            print(f"\n  批次 {batch_idx + 1}/{num_batches} (个体 {start_idx+1}-{end_idx}):")
            
            # 为每个机器人分配神经网络
            for i in range(batch_size):
                individual_idx = start_idx + i
                self.robot_controllers[i].set_neural_network(population[individual_idx])
            
            # 重置机器人位置
            self.reset_robots_positions()
            
            # 并行运行所有机器人
            steps = int(self.simulation_time * 1000 / self.timestep)
            
            for step in range(steps):
                # 所有机器人同时执行一步
                for i in range(batch_size):
                    self.robot_controllers[i].step()
                
                # Supervisor执行一步
                if self.supervisor.step(self.timestep) == -1:
                    break
                
                # 每100步显示一次进度
                if step % 100 == 0 and step > 0:
                    progress = (step / steps) * 100
                    print(f"    进度: {progress:.1f}%", end='\r')
            
            print(f"    进度: 100.0%")
            
            # 收集适应度分数
            for i in range(batch_size):
                individual_idx = start_idx + i
                fitness = self.robot_controllers[i].get_fitness()
                fitness_scores[individual_idx] = fitness
                
                stats = self.robot_controllers[i].get_stats()
                print(f"    个体{individual_idx+1}: 适应度={fitness:.2f}, " +
                      f"距离={stats['distance']:.2f}m, " +
                      f"碰撞={stats['collisions']}, " +
                      f"完成={'是' if stats['completed_lap'] else '否'}")
        
        return fitness_scores
    
    def train(self):
        """执行完整的训练过程"""
        print("\n" + "=" * 70)
        print("开始并行训练")
        print("=" * 70)
        
        start_time = time.time()
        
        # 初始化种群
        self.ga.initialize_population()
        
        # 进化循环
        for generation in range(self.generations):
            print(f"\n{'='*70}")
            print(f"第 {generation + 1}/{self.generations} 代")
            print(f"{'='*70}")
            
            # 并行评估种群
            fitness_scores = self.evaluate_population_parallel()
            
            # 更新遗传算法
            self.ga.evaluate_population(fitness_scores)
            
            # 获取统计信息
            stats = self.ga.get_statistics()
            
            # 记录日志
            self.training_log['generations'].append(generation + 1)
            self.training_log['best_fitness'].append(stats['best_fitness'])
            self.training_log['avg_fitness'].append(stats['avg_fitness_history'][-1])
            
            # 每10代保存一次检查点
            if (generation + 1) % 10 == 0:
                save_checkpoint(self.ga, generation + 1)
            
            # 如果不是最后一代，进化到下一代
            if generation < self.generations - 1:
                self.ga.evolve()
        
        # 训练结束
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "=" * 70)
        print("训练完成!")
        print("=" * 70)
        print(f"总训练时间: {training_time:.2f}秒 ({training_time/60:.2f}分钟)")
        print(f"最佳适应度: {self.ga.best_fitness:.2f}")
        
        # 保存最佳模型
        best_individual = self.ga.get_best_individual()
        if best_individual:
            metadata = {
                'generations': self.generations,
                'best_fitness': self.ga.best_fitness,
                'training_time': training_time,
                'population_size': self.ga.population_size,
                'parallel_robots': self.num_robots
            }
            save_model(best_individual, metadata=metadata)
        
        # 保存训练日志
        save_training_log(self.training_log)
        
        # 打印最终统计
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        print("\n" + "=" * 70)
        print("训练统计")
        print("=" * 70)
        
        stats = self.ga.get_statistics()
        
        print(f"总代数: {stats['generation']}")
        print(f"最佳适应度: {stats['best_fitness']:.2f}")
        print(f"初始平均适应度: {stats['avg_fitness_history'][0]:.2f}")
        print(f"最终平均适应度: {stats['avg_fitness_history'][-1]:.2f}")
        print(f"适应度提升: {stats['avg_fitness_history'][-1] - stats['avg_fitness_history'][0]:.2f}")
        print(f"并行加速比: {self.num_robots}x")


def main():
    """主函数"""
    try:
        # 创建并行训练器
        # 可以指定机器人数量，例如: ParallelTrainer(num_robots=10)
        trainer = ParallelTrainer()
        
        # 开始训练
        trainer.train()
        
        print("\n训练成功完成！")
        print("最佳模型已保存到 models/best_model.pkl")
        print("可以运行 test.py 来测试训练好的模型")
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
