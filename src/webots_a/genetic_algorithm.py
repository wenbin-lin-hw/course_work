"""
遗传算法核心类 - 进化的引擎

这是整个系统的核心！遗传算法通过模拟自然进化来优化机器人的"大脑"

工作流程：
1. 初始化种群：创建N个随机的"大脑"（神经网络）
2. 评估适应度：让每个机器人在环境中运行，评估表现
3. 选择：选出表现好的机器人作为"父母"
4. 交叉：父母的基因组合产生"孩子"
5. 变异：随机改变一些基因，增加多样性
6. 重复2-5步，直到找到最优解

关键概念：
- 基因 = 神经网络的权重
- 适应度 = 机器人的表现分数
- 进化 = 好的基因被保留和传播，坏的基因被淘汰
"""
import numpy as np
from neural_network import NeuralNetwork
from config import GA_CONFIG


class GeneticAlgorithm:
    """
    遗传算法类

    管理整个进化过程
    """

    def __init__(self, config=None):
        """
        初始化遗传算法

        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config if config else GA_CONFIG

        # 种群参数
        self.population_size = self.config['population_size']
        self.elite_size = self.config['elite_size']

        # 遗传操作参数
        self.mutation_rate = self.config['mutation_rate']
        self.crossover_rate = self.config['crossover_rate']
        self.mutation_strength = self.config['mutation_strength']

        # 神经网络参数
        self.input_size = self.config['input_size']
        self.hidden_size = self.config['hidden_size']
        self.output_size = self.config['output_size']

        # 初始化种群
        self.population = []
        self.fitness_scores = []
        self.generation = 0

        # 统计信息
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = -float('inf')

    def initialize_population(self):
        """
        初始化种群

        创建N个随机的神经网络作为初始种群
        这些随机的"大脑"是进化的起点
        """
        print(f"初始化种群：{self.population_size}个个体")
        self.population = []

        for i in range(self.population_size):
            # 创建一个随机的神经网络
            nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            self.population.append(nn)

        self.fitness_scores = [0.0] * self.population_size
        print(f"种群初始化完成，每个个体有{self.population[0].get_weights_count()}个基因")

    def evaluate_population(self, fitness_scores):
        """
        评估种群适应度

        接收每个个体的适应度分数，更新种群状态

        Args:
            fitness_scores: 列表，每个个体的适应度分数
        """
        self.fitness_scores = fitness_scores

        # 更新最佳个体
        max_fitness = max(fitness_scores)
        if max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
            best_idx = fitness_scores.index(max_fitness)
            self.best_individual = self.population[best_idx].copy()

        # 记录统计信息
        self.best_fitness_history.append(max_fitness)
        self.avg_fitness_history.append(np.mean(fitness_scores))

        print(f"\n第{self.generation}代:")
        print(f"  最佳适应度: {max_fitness:.2f}")
        print(f"  平均适应度: {np.mean(fitness_scores):.2f}")
        print(f"  最差适应度: {min(fitness_scores):.2f}")

    def select_parents(self):
        """
        选择父母

        使用轮盘赌选择法（Roulette Wheel Selection）
        适应度越高的个体，被选中的概率越大

        Returns:
            list: 选中的父母个体索引
        """
        # 确保所有适应度非负
        min_fitness = min(self.fitness_scores)
        adjusted_fitness = [f - min_fitness + 1e-6 for f in self.fitness_scores]

        # 计算选择概率
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]

        # 选择父母（数量等于种群大小）
        parent_indices = np.random.choice(
            self.population_size,
            size=self.population_size - self.elite_size,
            p=probabilities,
            replace=True
        )

        return parent_indices.tolist()

    def crossover(self, parent1, parent2):
        """
        交叉操作（基因重组）

        将两个父母的基因组合，产生一个孩子
        使用单点交叉法：随机选择一个点，前半部分来自父母1，后半部分来自父母2

        Args:
            parent1: 父母1的神经网络
            parent2: 父母2的神经网络

        Returns:
            NeuralNetwork: 孩子的神经网络
        """
        # 获取父母的基因
        genes1 = parent1.get_weights()
        genes2 = parent2.get_weights()

        # 创建孩子
        child = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)

        # 决定是否进行交叉
        if np.random.random() < self.crossover_rate:
            # 单点交叉
            crossover_point = np.random.randint(1, len(genes1))
            child_genes = np.concatenate([
                genes1[:crossover_point],
                genes2[crossover_point:]
            ])
        else:
            # 不交叉，随机选择一个父母
            child_genes = genes1.copy() if np.random.random() < 0.5 else genes2.copy()

        child.set_weights(child_genes)
        return child

    def mutate(self, individual):
        """
        变异操作

        随机改变一些基因，增加种群多样性
        这防止算法陷入局部最优

        Args:
            individual: 要变异的神经网络

        Returns:
            NeuralNetwork: 变异后的神经网络
        """
        genes = individual.get_weights()

        # 对每个基因，以mutation_rate的概率进行变异
        for i in range(len(genes)):
            if np.random.random() < self.mutation_rate:
                # 添加随机噪声
                genes[i] += np.random.randn() * self.mutation_strength

        individual.set_weights(genes)
        return individual

    def evolve(self):
        """
        进化到下一代

        这是遗传算法的核心流程：
        1. 保留精英（最好的几个个体直接进入下一代）
        2. 选择父母
        3. 交叉产生孩子
        4. 变异
        5. 组成新一代种群

        Returns:
            list: 新一代种群
        """
        print(f"\n开始进化到第{self.generation + 1}代...")

        # 1. 精英选择：保留最好的几个个体
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        elites = [self.population[i].copy() for i in elite_indices]
        print(f"  保留{self.elite_size}个精英个体")

        # 2. 选择父母
        parent_indices = self.select_parents()
        print(f"  选择{len(parent_indices)}对父母")

        # 3. 交叉和变异产生新个体
        new_population = elites.copy()

        for i in range(0, len(parent_indices), 2):
            # 选择两个父母
            parent1_idx = parent_indices[i]
            parent2_idx = parent_indices[min(i + 1, len(parent_indices) - 1)]

            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            # 交叉产生孩子
            child = self.crossover(parent1, parent2)

            # 变异
            child = self.mutate(child)

            new_population.append(child)

        # 确保种群大小不变
        new_population = new_population[:self.population_size]

        # 更新种群
        self.population = new_population
        self.generation += 1

        print(f"  进化完成，新一代有{len(self.population)}个个体")

        return self.population

    def get_best_individual(self):
        """
        获取最佳个体

        Returns:
            NeuralNetwork: 历史最佳个体
        """
        return self.best_individual

    def get_population(self):
        """
        获取当前种群

        Returns:
            list: 当前种群的所有个体
        """
        return self.population

    def get_statistics(self):
        """
        获取统计信息

        Returns:
            dict: 包含进化历史的统计信息
        """
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'population_size': self.population_size
        }
"""
遗传算法核心类 - 进化的引擎

这是整个系统的核心！遗传算法通过模拟自然进化来优化机器人的"大脑"

工作流程：
1. 初始化种群：创建N个随机的"大脑"（神经网络）
2. 评估适应度：让每个机器人在环境中运行，评估表现
3. 选择：选出表现好的机器人作为"父母"
4. 交叉：父母的基因组合产生"孩子"
5. 变异：随机改变一些基因，增加多样性
6. 重复2-5步，直到找到最优解

关键概念：
- 基因 = 神经网络的权重
- 适应度 = 机器人的表现分数
- 进化 = 好的基因被保留和传播，坏的基因被淘汰
"""
import numpy as np
from neural_network import NeuralNetwork
from config import GA_CONFIG


class GeneticAlgorithm:
    """
    遗传算法类
    
    管理整个进化过程
    """
    
    def __init__(self, config=None):
        """
        初始化遗传算法
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config if config else GA_CONFIG
        
        # 种群参数
        self.population_size = self.config['population_size']
        self.elite_size = self.config['elite_size']
        
        # 遗传操作参数
        self.mutation_rate = self.config['mutation_rate']
        self.crossover_rate = self.config['crossover_rate']
        self.mutation_strength = self.config['mutation_strength']
        
        # 神经网络参数
        self.input_size = self.config['input_size']
        self.hidden_size = self.config['hidden_size']
        self.output_size = self.config['output_size']
        
        # 初始化种群
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        
        # 统计信息
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = -float('inf')
    
    def initialize_population(self):
        """
        初始化种群
        
        创建N个随机的神经网络作为初始种群
        这些随机的"大脑"是进化的起点
        """
        print(f"初始化种群：{self.population_size}个个体")
        self.population = []
        
        for i in range(self.population_size):
            # 创建一个随机的神经网络
            nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            self.population.append(nn)
        
        self.fitness_scores = [0.0] * self.population_size
        print(f"种群初始化完成，每个个体有{self.population[0].get_weights_count()}个基因")
    
    def evaluate_population(self, fitness_scores):
        """
        评估种群适应度
        
        接收每个个体的适应度分数，更新种群状态
        
        Args:
            fitness_scores: 列表，每个个体的适应度分数
        """
        self.fitness_scores = fitness_scores
        
        # 更新最佳个体
        max_fitness = max(fitness_scores)
        if max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
            best_idx = fitness_scores.index(max_fitness)
            self.best_individual = self.population[best_idx].copy()
        
        # 记录统计信息
        self.best_fitness_history.append(max_fitness)
        self.avg_fitness_history.append(np.mean(fitness_scores))
        
        print(f"\n第{self.generation}代:")
        print(f"  最佳适应度: {max_fitness:.2f}")
        print(f"  平均适应度: {np.mean(fitness_scores):.2f}")
        print(f"  最差适应度: {min(fitness_scores):.2f}")
    
    def select_parents(self):
        """
        选择父母
        
        使用轮盘赌选择法（Roulette Wheel Selection）
        适应度越高的个体，被选中的概率越大
        
        Returns:
            list: 选中的父母个体索引
        """
        # 确保所有适应度非负
        min_fitness = min(self.fitness_scores)
        adjusted_fitness = [f - min_fitness + 1e-6 for f in self.fitness_scores]
        
        # 计算选择概率
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]
        
        # 选择父母（数量等于种群大小）
        parent_indices = np.random.choice(
            self.population_size,
            size=self.population_size - self.elite_size,
            p=probabilities,
            replace=True
        )
        
        return parent_indices.tolist()
    
    def crossover(self, parent1, parent2):
        """
        交叉操作（基因重组）
        
        将两个父母的基因组合，产生一个孩子
        使用单点交叉法：随机选择一个点，前半部分来自父母1，后半部分来自父母2
        
        Args:
            parent1: 父母1的神经网络
            parent2: 父母2的神经网络
            
        Returns:
            NeuralNetwork: 孩子的神经网络
        """
        # 获取父母的基因
        genes1 = parent1.get_weights()
        genes2 = parent2.get_weights()
        
        # 创建孩子
        child = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        
        # 决定是否进行交叉
        if np.random.random() < self.crossover_rate:
            # 单点交叉
            crossover_point = np.random.randint(1, len(genes1))
            child_genes = np.concatenate([
                genes1[:crossover_point],
                genes2[crossover_point:]
            ])
        else:
            # 不交叉，随机选择一个父母
            child_genes = genes1.copy() if np.random.random() < 0.5 else genes2.copy()
        
        child.set_weights(child_genes)
        return child
    
    def mutate(self, individual):
        """
        变异操作
        
        随机改变一些基因，增加种群多样性
        这防止算法陷入局部最优
        
        Args:
            individual: 要变异的神经网络
            
        Returns:
            NeuralNetwork: 变异后的神经网络
        """
        genes = individual.get_weights()
        
        # 对每个基因，以mutation_rate的概率进行变异
        for i in range(len(genes)):
            if np.random.random() < self.mutation_rate:
                # 添加随机噪声
                genes[i] += np.random.randn() * self.mutation_strength
        
        individual.set_weights(genes)
        return individual
    
    def evolve(self):
        """
        进化到下一代
        
        这是遗传算法的核心流程：
        1. 保留精英（最好的几个个体直接进入下一代）
        2. 选择父母
        3. 交叉产生孩子
        4. 变异
        5. 组成新一代种群
        
        Returns:
            list: 新一代种群
        """
        print(f"\n开始进化到第{self.generation + 1}代...")
        
        # 1. 精英选择：保留最好的几个个体
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        elites = [self.population[i].copy() for i in elite_indices]
        print(f"  保留{self.elite_size}个精英个体")
        
        # 2. 选择父母
        parent_indices = self.select_parents()
        print(f"  选择{len(parent_indices)}对父母")
        
        # 3. 交叉和变异产生新个体
        new_population = elites.copy()
        
        for i in range(0, len(parent_indices), 2):
            # 选择两个父母
            parent1_idx = parent_indices[i]
            parent2_idx = parent_indices[min(i + 1, len(parent_indices) - 1)]
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # 交叉产生孩子
            child = self.crossover(parent1, parent2)
            
            # 变异
            child = self.mutate(child)
            
            new_population.append(child)
        
        # 确保种群大小不变
        new_population = new_population[:self.population_size]
        
        # 更新种群
        self.population = new_population
        self.generation += 1
        
        print(f"  进化完成，新一代有{len(self.population)}个个体")
        
        return self.population
    
    def get_best_individual(self):
        """
        获取最佳个体
        
        Returns:
            NeuralNetwork: 历史最佳个体
        """
        return self.best_individual
    
    def get_population(self):
        """
        获取当前种群
        
        Returns:
            list: 当前种群的所有个体
        """
        return self.population
    
    def get_statistics(self):
        """
        获取统计信息
        
        Returns:
            dict: 包含进化历史的统计信息
        """
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'population_size': self.population_size
        }
