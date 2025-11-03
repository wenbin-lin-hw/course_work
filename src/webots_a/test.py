"""
测试脚本 - 测试训练好的模型

这个脚本用于：
1. 加载训练好的最佳模型
2. 让机器人在环境中运行
3. 计算完成一圈的时间
4. 显示详细的性能统计

使用方法：
    在Webots中运行此脚本作为控制器
"""
import sys
import time
from robot_controller import EPuckController
from model_utils import load_model, list_saved_models, get_model_info
from config import TEST_CONFIG


class Tester:
    """
    测试器类
    
    管理测试过程
    """
    
    def __init__(self, model_path=None):
        """
        初始化测试器
        
        Args:
            model_path: 模型文件路径，如果为None则使用最佳模型
        """
        print("=" * 60)
        print("模型测试器 - E-puck循迹避障")
        print("=" * 60)
        
        # 加载模型
        try:
            self.neural_network, self.metadata = load_model(model_path)
            print("\n模型加载成功!")
            
            if self.metadata:
                print("\n模型信息:")
                for key, value in self.metadata.items():
                    print(f"  {key}: {value}")
        
        except FileNotFoundError:
            print("\n错误: 未找到训练好的模型!")
            print("请先运行 train.py 进行训练")
            
            # 列出可用的模型
            models = list_saved_models()
            if models:
                print("\n可用的模型:")
                for model in models:
                    info = get_model_info(model)
                    print(f"  - {model}")
                    print(f"    时间: {info.get('timestamp', 'Unknown')}")
                    print(f"    适应度: {info.get('metadata', {}).get('best_fitness', 'Unknown')}")
            
            sys.exit(1)
        
        # 初始化机器人控制器
        self.controller = EPuckController(self.neural_network)
        
        # 测试参数
        self.max_test_time = TEST_CONFIG['max_test_time']
        self.num_laps = TEST_CONFIG['num_laps']
        
        print(f"\n测试配置:")
        print(f"  最大测试时间: {self.max_test_time}秒")
        print(f"  目标圈数: {self.num_laps}")
        print()
    
    def test_single_lap(self):
        """
        测试单圈性能
        
        Returns:
            dict: 测试结果
        """
        print("\n" + "=" * 60)
        print("开始测试 - 单圈性能")
        print("=" * 60)
        
        # 重置控制器
        self.controller.reset()
        
        # 记录起始时间和位置
        start_time = time.time()
        start_position = None
        
        # 运行机器人
        steps = 0
        max_steps = int(self.max_test_time * 1000 / self.controller.timestep)
        
        completed = False
        lap_time = None
        
        print("\n机器人开始运行...")
        print("(按Ctrl+C可以提前停止)")
        
        try:
            while steps < max_steps:
                # 执行一步
                if not self.controller.step():
                    print("\n仿真结束")
                    break
                
                steps += 1
                
                # 记录起始位置
                if start_position is None:
                    start_position = self.controller.get_position()
                    if start_position:
                        print(f"起始位置: ({start_position[0]:.3f}, {start_position[1]:.3f})")
                
                # 检查是否完成一圈
                if self.controller.fitness_evaluator.completed_lap:
                    completed = True
                    lap_time = time.time() - start_time
                    print(f"\n完成一圈! 用时: {lap_time:.2f}秒")
                    break
                
                # 每100步显示一次进度
                if steps % 100 == 0:
                    current_stats = self.controller.get_stats()
                    print(f"  步数: {steps}, 距离: {current_stats['distance']:.2f}m, " +
                          f"碰撞: {current_stats['collisions']}")
        
        except KeyboardInterrupt:
            print("\n\n测试被用户中断")
        
        # 获取最终统计
        final_stats = self.controller.get_stats()
        
        # 如果完成了一圈，计算准确的圈时间
        if completed and lap_time is None:
            lap_time = final_stats['steps'] * self.controller.timestep / 1000.0
        
        # 打印结果
        self.print_test_results(final_stats, lap_time, completed)
        
        return {
            'completed': completed,
            'lap_time': lap_time,
            'stats': final_stats
        }
    
    def print_test_results(self, stats, lap_time, completed):
        """
        打印测试结果
        
        Args:
            stats: 统计信息字典
            lap_time: 圈时间（秒）
            completed: 是否完成
        """
        print("\n" + "=" * 60)
        print("测试结果")
        print("=" * 60)
        
        print(f"\n完成状态: {'✓ 完成一圈' if completed else '✗ 未完成'}")
        
        if completed and lap_time:
            print(f"\n⏱️  圈时间: {lap_time:.2f}秒")
            print(f"   平均速度: {stats['distance'] / lap_time:.3f} m/s")
        
        print(f"\n📊 性能指标:")
        print(f"   总距离: {stats['distance']:.2f}米")
        print(f"   平均速度: {stats['avg_speed']:.3f}")
        print(f"   循迹得分: {stats['line_following']:.3f}")
        print(f"   避障得分: {stats['obstacle_avoidance']:.3f}")
        print(f"   运动平滑度: {stats['smoothness']:.3f}")
        
        print(f"\n⚠️  问题统计:")
        print(f"   碰撞次数: {stats['collisions']}")
        print(f"   偏离次数: {stats['deviations']}")
        
        print(f"\n🎯 适应度分数: {stats['fitness']:.2f}")
        
        # 评级
        if completed:
            if lap_time < 30:
                rating = "优秀 ⭐⭐⭐"
            elif lap_time < 60:
                rating = "良好 ⭐⭐"
            else:
                rating = "及格 ⭐"
        else:
            rating = "需要改进"
        
        print(f"\n总体评价: {rating}")
    
    def test_multiple_laps(self):
        """
        测试多圈性能
        
        Returns:
            list: 每圈的测试结果
        """
        print("\n" + "=" * 60)
        print(f"开始测试 - {self.num_laps}圈性能")
        print("=" * 60)
        
        results = []
        
        for lap in range(self.num_laps):
            print(f"\n第 {lap + 1}/{self.num_laps} 圈")
            result = self.test_single_lap()
            results.append(result)
            
            if not result['completed']:
                print(f"\n第{lap + 1}圈未完成，停止测试")
                break
        
        # 打印总结
        if len(results) > 1:
            self.print_multiple_laps_summary(results)
        
        return results
    
    def print_multiple_laps_summary(self, results):
        """
        打印多圈测试总结
        
        Args:
            results: 测试结果列表
        """
        print("\n" + "=" * 60)
        print("多圈测试总结")
        print("=" * 60)
        
        completed_laps = [r for r in results if r['completed']]
        
        if completed_laps:
            lap_times = [r['lap_time'] for r in completed_laps]
            
            print(f"\n完成圈数: {len(completed_laps)}/{len(results)}")
            print(f"\n圈时间统计:")
            print(f"  最快: {min(lap_times):.2f}秒")
            print(f"  最慢: {max(lap_times):.2f}秒")
            print(f"  平均: {sum(lap_times)/len(lap_times):.2f}秒")
            
            print(f"\n各圈时间:")
            for i, lap_time in enumerate(lap_times):
                print(f"  第{i+1}圈: {lap_time:.2f}秒")
        else:
            print("\n没有完成任何一圈")
    
    def continuous_test(self):
        """
        连续测试模式
        
        让机器人持续运行，直到用户停止
        """
        print("\n" + "=" * 60)
        print("连续测试模式")
        print("=" * 60)
        print("\n机器人将持续运行，按Ctrl+C停止")
        
        self.controller.reset()
        
        try:
            while True:
                if not self.controller.step():
                    break
        
        except KeyboardInterrupt:
            print("\n\n测试停止")
        
        # 显示统计
        stats = self.controller.get_stats()
        self.print_test_results(stats, None, stats['completed_lap'])


def main():
    """主函数"""
    print("\n选择测试模式:")
    print("1. 单圈测试（默认）")
    print("2. 多圈测试")
    print("3. 连续测试")
    
    # 如果在Webots中运行，直接使用单圈测试
    # 如果需要交互，可以取消下面的注释
    # choice = input("\n请选择 (1/2/3): ").strip()
    choice = "1"  # 默认单圈测试
    
    try:
        # 创建测试器
        tester = Tester()
        
        # 根据选择执行测试
        if choice == "2":
            tester.test_multiple_laps()
        elif choice == "3":
            tester.continuous_test()
        else:
            tester.test_single_lap()
        
        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
