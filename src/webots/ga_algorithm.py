"""
遗传算法核心模块
用于进化 e-puck 机器人的控制器参数
"""

import numpy as np
import random
import json
import os
from datetime import datetime


class GeneticAlgorithm:
    """遗传算法类"""

    def __init__(self,
                 population_size=20,
                 genome_size=24,  # 8个传感器 * 3个权重层
                 mutation_rate=0.1,
                 crossover_rate=0.7,
                 elite_size=2):
        """
        初始化遗传算法

        Args:
            population_size: 种群大小
            genome_size: 基因组大小（神经网络权重数量）
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elite_size: 精英个体数量
        """
        self.population_size = population_size
        self.genome_size = genome_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

        # 初始化种群
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.best_genome = None
        self.best_fitness = -float('inf')
        self.fitness_history = []

        self._initialize_population()

    def _initialize_population(self):
        """初始化种群（随机生成）"""
        self.population = []
        for _ in range(self.population_size):
            # 生成随机基因组（权重范围 -1 到 1）
            genome = np.random.uniform(-1, 1, self.genome_size)
            self.population.append(genome)
        print(f"初始化种群: {self.population_size} 个体")

    def decode_genome(self, genome):
        """
        解码基因组为神经网络权重

        网络结构:
        - 输入层: 8个距离传感器 + 3个地面传感器 = 11个输入
        - 隐藏层: 6个神经元
        - 输出层: 2个输出（左轮速度，右轮速度）

        Args:
            genome: 基因组数组

        Returns:
            weights: 字典包含各层权重
        """
        idx = 0
        weights = {}

        # 输入层到隐藏层的权重 (11 * 6 = 66)
        input_size = 11
        hidden_size = 6
        w1_size = input_size * hidden_size

        if len(genome) < w1_size:
            # 如果基因组太小，扩展它
            genome = np.pad(genome, (0, w1_size - len(genome)), 'constant')

        weights['w1'] = genome[idx:idx + w1_size].reshape(input_size, hidden_size)
        idx += w1_size

        # 隐藏层到输出层的权重 (6 * 2 = 12)
        output_size = 2
        w2_size = hidden_size * output_size

        if len(genome) < idx + w2_size:
            genome = np.pad(genome, (0, idx + w2_size - len(genome)), 'constant')

        weights['w2'] = genome[idx:idx + w2_size].reshape(hidden_size, output_size)

        return weights

    def evaluate_fitness(self, individual_idx, fitness_value):
        """
        评估个体适应度

        Args:
            individual_idx: 个体索引
            fitness_value: 适应度值
        """
        if individual_idx < len(self.fitness_scores):
            self.fitness_scores[individual_idx] = fitness_value
        else:
            self.fitness_scores.append(fitness_value)

        # 更新最佳个体
        if fitness_value > self.best_fitness:
            self.best_fitness = fitness_value
            self.best_genome = self.population[individual_idx].copy()
            print(f"🏆 新的最佳适应度: {self.best_fitness:.2f}")

    def selection(self):
        """
        选择操作（锦标赛选择）

        Returns:
            selected: 选中的父代索引列表
        """
        selected = []

        # 保留精英个体
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        selected.extend(elite_indices)

        # 锦标赛选择其余个体
        tournament_size = 3
        while len(selected) < self.population_size:
            # 随机选择tournament_size个个体
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]

            # 选择适应度最高的
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(winner_idx)

        return selected

    def crossover(self, parent1, parent2):
        """
        交叉操作（单点交叉）

        Args:
            parent1: 父代1基因组
            parent2: 父代2基因组

        Returns:
            child1, child2: 两个子代基因组
        """
        if random.random() < self.crossover_rate:
            # 单点交叉
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        else:
            # 不交叉，直接复制
            child1 = parent1.copy()
            child2 = parent2.copy()

        return child1, child2

    def mutate(self, genome):
        """
        变异操作（高斯变异）

        Args:
            genome: 基因组

        Returns:
            mutated_genome: 变异后的基因组
        """
        mutated_genome = genome.copy()

        for i in range(len(mutated_genome)):
            if random.random() < self.mutation_rate:
                # 高斯变异
                mutation = np.random.normal(0, 0.3)
                mutated_genome[i] += mutation
                # 限制范围在 [-1, 1]
                mutated_genome[i] = np.clip(mutated_genome[i], -1, 1)

        return mutated_genome

    def evolve(self):
        """
        进化到下一代
        """
        print(f"\n{'=' * 60}")
        print(f"第 {self.generation} 代进化")
        print(f"{'=' * 60}")

        # 统计当前代
        avg_fitness = np.mean(self.fitness_scores)
        max_fitness = np.max(self.fitness_scores)
        min_fitness = np.min(self.fitness_scores)

        print(f"适应度统计:")
        print(f"  平均: {avg_fitness:.2f}")
        print(f"  最大: {max_fitness:.2f}")
        print(f"  最小: {min_fitness:.2f}")
        print(f"  历史最佳: {self.best_fitness:.2f}")

        self.fitness_history.append({
            'generation': self.generation,
            'avg': avg_fitness,
            'max': max_fitness,
            'min': min_fitness,
            'best_ever': self.best_fitness
        })

        # 选择
        selected_indices = self.selection()

        # 创建新种群
        new_population = []

        # 保留精英
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())

        # 交叉和变异生成其余个体
        while len(new_population) < self.population_size:
            # 随机选择两个父代
            parent1_idx = random.choice(selected_indices)
            parent2_idx = random.choice(selected_indices)

            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            # 交叉
            child1, child2 = self.crossover(parent1, parent2)

            # 变异
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        # 更新种群
        self.population = new_population
        self.fitness_scores = []
        self.generation += 1

        print(f"✓ 进化完成，进入第 {self.generation} 代")

    def save_best_genome(self, filepath='best_genome.json'):
        """
        保存最佳基因组

        Args:
            filepath: 保存路径
        """
        if self.best_genome is None:
            print("⚠ 没有最佳基因组可保存")
            return

        # 创建目录
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        data = {
            'genome': self.best_genome.tolist(),
            'fitness': float(self.best_fitness),
            'generation': self.generation,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'population_size': self.population_size,
                'genome_size': self.genome_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ 最佳基因组已保存到: {filepath}")
        print(f"  适应度: {self.best_fitness:.2f}")
        print(f"  代数: {self.generation}")

    def load_genome(self, filepath='best_genome.json'):
        """
        加载保存的基因组

        Args:
            filepath: 文件路径

        Returns:
            genome: 加载的基因组，如果失败返回None
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            genome = np.array(data['genome'])
            print(f"✓ 成功加载基因组: {filepath}")
            print(f"  适应度: {data['fitness']:.2f}")
            print(f"  代数: {data['generation']}")

            return genome
        except Exception as e:
            print(f"✗ 加载基因组失败: {e}")
            return None

    def save_training_history(self, filepath='training_history.json'):
        """
        保存训练历史

        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.fitness_history, f, indent=2)

        print(f"✓ 训练历史已保存到: {filepath}")

    def get_current_genome(self, individual_idx):
        """
        获取当前个体的基因组

        Args:
            individual_idx: 个体索引

        Returns:
            genome: 基因组数组
        """
        if individual_idx < len(self.population):
            return self.population[individual_idx]
        return None


class FitnessCalculator:
    """适应度计算器"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置计数器"""
        self.distance_traveled = 0.0
        self.time_on_track = 0.0
        self.time_off_track = 0.0
        self.collisions = 0
        self.lap_completed = False
        self.total_time = 0.0
        self.last_position = None
        self.track_following_score = 0.0
        self.obstacle_avoidance_score = 0.0

    def update(self, position, on_track, collision, dt):
        """
        更新适应度相关数据

        Args:
            position: 机器人位置 [x, y, z]
            on_track: 是否在赛道上
            collision: 是否发生碰撞
            dt: 时间步长（秒）
        """
        # 计算移动距离
        if self.last_position is not None:
            dx = position[0] - self.last_position[0]
            dy = position[1] - self.last_position[1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            self.distance_traveled += distance

        self.last_position = position

        # 更新时间
        self.total_time += dt

        if on_track:
            self.time_on_track += dt
            self.track_following_score += 1.0
        else:
            self.time_off_track += dt
            self.track_following_score -= 0.5

        if collision:
            self.collisions += 1
            self.obstacle_avoidance_score -= 10.0
        else:
            self.obstacle_avoidance_score += 0.1

    def calculate_fitness(self):
        """
        计算最终适应度

        Returns:
            fitness: 适应度值
        """
        # 基础分数：移动距离
        distance_score = self.distance_traveled * 10.0

        # 赛道跟随分数
        if self.total_time > 0:
            track_ratio = self.time_on_track / self.total_time
            track_score = track_ratio * 100.0
        else:
            track_score = 0.0

        # 碰撞惩罚
        collision_penalty = self.collisions * 50.0

        # 完成圈数奖励
        lap_bonus = 500.0 if self.lap_completed else 0.0

        # 综合适应度
        fitness = (
                distance_score +
                track_score +
                self.track_following_score +
                self.obstacle_avoidance_score +
                lap_bonus -
                collision_penalty
        )

        return max(0, fitness)  # 确保非负

    def get_statistics(self):
        """
        获取统计信息

        Returns:
            stats: 统计字典
        """
        return {
            'distance': self.distance_traveled,
            'time_on_track': self.time_on_track,
            'time_off_track': self.time_off_track,
            'collisions': self.collisions,
            'lap_completed': self.lap_completed,
            'total_time': self.total_time,
            'fitness': self.calculate_fitness()
        }
