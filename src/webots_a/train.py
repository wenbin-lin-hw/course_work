"""
训练脚本 - 使用遗传算法训练机器人

这是主训练脚本，执行完整的进化过程：
1. 初始化遗传算法和种群
2. 对每一代：
   - 评估每个个体的适应度
   - 选择、交叉、变异产生新一代
3. 保存最佳模型

使用方法：
    在Webots中运行此脚本作为控制器
"""
import sys
import time
from genetic_algorithm import GeneticAlgorithm
from robot_controller import EPuckController
from model_utils import save_model, save_checkpoint, save_training_log
from config import GA_CONFIG


class Trainer:
    """
    训练器类
    
    管理整个训练过程
    """
    
    def __init__(self):
        """初始化训练器"""
        print("=" * 60)
        print("遗传算法训练器 - E-puck循迹避障")
        print("=" * 60)
        
        # 初始化遗传算法
        self.ga = GeneticAlgorithm(GA_CONFIG)
        
        # 初始化机器人控制器
        self.controller = EPuckController()
        
        # 训练参数
        self.generations = GA_CONFIG['generations']
        self.simulation_time = GA_CONFIG['simulation_time']
        
        # 训练日志
        self.training_log = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_individual_stats': []
        }
        
        print(f"\n配置:")
        print(f"  种群大小: {self.ga.population_size}")
        print(f"  进化代数: {self.generations}")
        print(f"  每次模拟时间: {self.simulation_time}秒")
        print(f"  神经网络结构: {self.ga.input_size}-{self.ga.hidden_size}-{self.ga.output_size}")
        print()
    
    def evaluate_individual(self, neural_network):
        """
        评估单个个体
        
        让机器人在环境中运行，计算适应度
        
        Args:
            neural_network: 要评估的神经网络
            
        Returns:
            float: 适应度分数
        """
        # 设置神经网络
        self.controller.set_neural_network(neural_network)
        
        # 重置控制器
        self.controller.reset()
        
        # 运行机器人
        fitness = self.controller.run(self.simulation_time)
        
        return fitness
    
    def evaluate_population(self):
        """
        评估整个种群
        
        Returns:
            list: 每个个体的适应度分数
        """
        print(f"\n评估第{self.ga.generation}代种群...")
        
        population = self.ga.get_population()
        fitness_scores = []
        
        for i, individual in enumerate(population):
            print(f"  评估个体 {i+1}/{len(population)}...", end=' ')
            
            # 评估个体
            fitness = self.evaluate_individual(individual)
            fitness_scores.append(fitness)
            
            # 获取详细统计
            stats = self.controller.get_stats()
            
            print(f"适应度: {fitness:.2f}, 距离: {stats['distance']:.2f}m, " +
                  f"碰撞: {stats['collisions']}, 完成: {stats['completed_lap']}")
        
        return fitness_scores
    
    def train(self):
        """
        执行完整的训练过程
        """
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        start_time = time.time()
        
        # 初始化种群
        self.ga.initialize_population()
        
        # 进化循环
        for generation in range(self.generations):
            print(f"\n{'='*60}")
            print(f"第 {generation + 1}/{self.generations} 代")
            print(f"{'='*60}")
            
            # 评估种群
            fitness_scores = self.evaluate_population()
            
            # 更新遗传算法
            self.ga.evaluate_population(fitness_scores)
            
            # 获取统计信息
            stats = self.ga.get_statistics()
            
            # 记录日志
            self.training_log['generations'].append(generation + 1)
            self.training_log['best_fitness'].append(stats['best_fitness'])
            self.training_log['avg_fitness'].append(stats['avg_fitness_history'][-1])
            
            # 获取最佳个体的详细统计
            best_individual = self.ga.get_best_individual()
            if best_individual:
                self.controller.set_neural_network(best_individual)
                self.controller.reset()
                self.controller.run(self.simulation_time)
                best_stats = self.controller.get_stats()
                self.training_log['best_individual_stats'].append(best_stats)
            
            # 每10代保存一次检查点
            if (generation + 1) % 10 == 0:
                save_checkpoint(self.ga, generation + 1)
            
            # 如果不是最后一代，进化到下一代
            if generation < self.generations - 1:
                self.ga.evolve()
        
        # 训练结束
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"总训练时间: {training_time:.2f}秒 ({training_time/60:.2f}分钟)")
        print(f"最佳适应度: {self.ga.best_fitness:.2f}")
        
        # 保存最佳模型
        best_individual = self.ga.get_best_individual()
        if best_individual:
            metadata = {
                'generations': self.generations,
                'best_fitness': self.ga.best_fitness,
                'training_time': training_time,
                'population_size': self.ga.population_size
            }
            save_model(best_individual, metadata=metadata)
        
        # 保存训练日志
        save_training_log(self.training_log)
        
        # 打印最终统计
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        print("\n" + "=" * 60)
        print("训练统计")
        print("=" * 60)
        
        stats = self.ga.get_statistics()
        
        print(f"总代数: {stats['generation']}")
        print(f"最佳适应度: {stats['best_fitness']:.2f}")
        print(f"初始平均适应度: {stats['avg_fitness_history'][0]:.2f}")
        print(f"最终平均适应度: {stats['avg_fitness_history'][-1]:.2f}")
        print(f"适应度提升: {stats['avg_fitness_history'][-1] - stats['avg_fitness_history'][0]:.2f}")
        
        # 最佳个体统计
        if self.training_log['best_individual_stats']:
            best_stats = self.training_log['best_individual_stats'][-1]
            print(f"\n最佳个体表现:")
            print(f"  行驶距离: {best_stats['distance']:.2f}m")
            print(f"  平均速度: {best_stats['avg_speed']:.2f}")
            print(f"  循迹得分: {best_stats['line_following']:.2f}")
            print(f"  避障得分: {best_stats['obstacle_avoidance']:.2f}")
            print(f"  碰撞次数: {best_stats['collisions']}")
            print(f"  完成一圈: {'是' if best_stats['completed_lap'] else '否'}")
            
            if best_stats['completed_lap']:
                lap_time = best_stats['steps'] * GA_CONFIG['time_step'] / 1000.0
                print(f"  完成时间: {lap_time:.2f}秒")


def main():
    """主函数"""
    try:
        # 创建训练器
        trainer = Trainer()
        
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
