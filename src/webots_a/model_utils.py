"""
模型保存和加载工具

提供保存和加载训练好的神经网络模型的功能
"""
import pickle
import os
import json
from datetime import datetime
from config import PATHS


def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建

    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")


def save_model(neural_network, filepath=None, metadata=None):
    """
    保存神经网络模型

    Args:
        neural_network: 要保存的神经网络
        filepath: 保存路径，如果为None则使用默认路径
        metadata: 元数据字典（可选），包含训练信息

    Returns:
        str: 保存的文件路径
    """
    if filepath is None:
        ensure_dir(PATHS['model_dir'])
        filepath = PATHS['best_model']

    # 准备保存的数据
    model_data = {
        'weights': neural_network.get_weights(),
        'input_size': neural_network.input_size,
        'hidden_size': neural_network.hidden_size,
        'output_size': neural_network.output_size,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metadata': metadata if metadata else {}
    }

    # 保存到文件
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"模型已保存到: {filepath}")

    # 同时保存一个可读的JSON文件（不包含权重）
    info_filepath = filepath.replace('.pkl', '_info.json')
    info_data = {
        'input_size': model_data['input_size'],
        'hidden_size': model_data['hidden_size'],
        'output_size': model_data['output_size'],
        'timestamp': model_data['timestamp'],
        'weights_count': len(model_data['weights']),
        'metadata': model_data['metadata']
    }

    with open(info_filepath, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=2, ensure_ascii=False)

    print(f"模型信息已保存到: {info_filepath}")

    return filepath


def load_model(filepath=None):
    """
    加载神经网络模型

    Args:
        filepath: 模型文件路径，如果为None则使用默认路径

    Returns:
        tuple: (neural_network, metadata) 神经网络和元数据
    """
    if filepath is None:
        filepath = PATHS['best_model']

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")

    # 从文件加载
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)

    # 重建神经网络
    from neural_network import NeuralNetwork
    neural_network = NeuralNetwork(
        model_data['input_size'],
        model_data['hidden_size'],
        model_data['output_size']
    )
    neural_network.set_weights(model_data['weights'])

    print(f"模型已加载: {filepath}")
    print(f"  输入层大小: {model_data['input_size']}")
    print(f"  隐藏层大小: {model_data['hidden_size']}")
    print(f"  输出层大小: {model_data['output_size']}")
    print(f"  保存时间: {model_data['timestamp']}")

    return neural_network, model_data.get('metadata', {})


def save_checkpoint(genetic_algorithm, generation, filepath=None):
    """
    保存训练检查点

    保存整个遗传算法的状态，可以用于恢复训练

    Args:
        genetic_algorithm: 遗传算法对象
        generation: 当前代数
        filepath: 保存路径

    Returns:
        str: 保存的文件路径
    """
    if filepath is None:
        ensure_dir(PATHS['model_dir'])
        filepath = PATHS['checkpoint'].format(generation)

    checkpoint_data = {
        'generation': generation,
        'population': [ind.get_weights() for ind in genetic_algorithm.population],
        'fitness_scores': genetic_algorithm.fitness_scores,
        'best_fitness': genetic_algorithm.best_fitness,
        'best_individual': genetic_algorithm.best_individual.get_weights() if genetic_algorithm.best_individual else None,
        'best_fitness_history': genetic_algorithm.best_fitness_history,
        'avg_fitness_history': genetic_algorithm.avg_fitness_history,
        'config': genetic_algorithm.config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    print(f"检查点已保存: {filepath}")

    return filepath


def load_checkpoint(filepath):
    """
    加载训练检查点

    Args:
        filepath: 检查点文件路径

    Returns:
        dict: 检查点数据
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")

    with open(filepath, 'rb') as f:
        checkpoint_data = pickle.load(f)

    print(f"检查点已加载: {filepath}")
    print(f"  代数: {checkpoint_data['generation']}")
    print(f"  最佳适应度: {checkpoint_data['best_fitness']:.2f}")

    return checkpoint_data


def save_training_log(log_data, filepath=None):
    """
    保存训练日志

    Args:
        log_data: 日志数据（字典或列表）
        filepath: 保存路径
    """
    if filepath is None:
        ensure_dir(PATHS['log_dir'])
        filepath = PATHS['log_file']

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"训练日志已保存: {filepath}")


def list_saved_models():
    """
    列出所有保存的模型

    Returns:
        list: 模型文件路径列表
    """
    model_dir = PATHS['model_dir']

    if not os.path.exists(model_dir):
        return []

    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(model_dir, filename)
            models.append(filepath)

    return sorted(models)


def get_model_info(filepath):
    """
    获取模型信息（不加载权重）

    Args:
        filepath: 模型文件路径

    Returns:
        dict: 模型信息
    """
    info_filepath = filepath.replace('.pkl', '_info.json')

    if os.path.exists(info_filepath):
        with open(info_filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 如果没有info文件，加载pkl文件获取信息
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)

    return {
        'input_size': model_data['input_size'],
        'hidden_size': model_data['hidden_size'],
        'output_size': model_data['output_size'],
        'timestamp': model_data.get('timestamp', 'Unknown'),
        'weights_count': len(model_data['weights']),
        'metadata': model_data.get('metadata', {})
    }
"""
模型保存和加载工具

提供保存和加载训练好的神经网络模型的功能
"""
import pickle
import os
import json
from datetime import datetime
from config import PATHS


def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")


def save_model(neural_network, filepath=None, metadata=None):
    """
    保存神经网络模型
    
    Args:
        neural_network: 要保存的神经网络
        filepath: 保存路径，如果为None则使用默认路径
        metadata: 元数据字典（可选），包含训练信息
        
    Returns:
        str: 保存的文件路径
    """
    if filepath is None:
        ensure_dir(PATHS['model_dir'])
        filepath = PATHS['best_model']
    
    # 准备保存的数据
    model_data = {
        'weights': neural_network.get_weights(),
        'input_size': neural_network.input_size,
        'hidden_size': neural_network.hidden_size,
        'output_size': neural_network.output_size,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metadata': metadata if metadata else {}
    }
    
    # 保存到文件
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"模型已保存到: {filepath}")
    
    # 同时保存一个可读的JSON文件（不包含权重）
    info_filepath = filepath.replace('.pkl', '_info.json')
    info_data = {
        'input_size': model_data['input_size'],
        'hidden_size': model_data['hidden_size'],
        'output_size': model_data['output_size'],
        'timestamp': model_data['timestamp'],
        'weights_count': len(model_data['weights']),
        'metadata': model_data['metadata']
    }
    
    with open(info_filepath, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=2, ensure_ascii=False)
    
    print(f"模型信息已保存到: {info_filepath}")
    
    return filepath


def load_model(filepath=None):
    """
    加载神经网络模型
    
    Args:
        filepath: 模型文件路径，如果为None则使用默认路径
        
    Returns:
        tuple: (neural_network, metadata) 神经网络和元数据
    """
    if filepath is None:
        filepath = PATHS['best_model']
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")
    
    # 从文件加载
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    # 重建神经网络
    from neural_network import NeuralNetwork
    neural_network = NeuralNetwork(
        model_data['input_size'],
        model_data['hidden_size'],
        model_data['output_size']
    )
    neural_network.set_weights(model_data['weights'])
    
    print(f"模型已加载: {filepath}")
    print(f"  输入层大小: {model_data['input_size']}")
    print(f"  隐藏层大小: {model_data['hidden_size']}")
    print(f"  输出层大小: {model_data['output_size']}")
    print(f"  保存时间: {model_data['timestamp']}")
    
    return neural_network, model_data.get('metadata', {})


def save_checkpoint(genetic_algorithm, generation, filepath=None):
    """
    保存训练检查点
    
    保存整个遗传算法的状态，可以用于恢复训练
    
    Args:
        genetic_algorithm: 遗传算法对象
        generation: 当前代数
        filepath: 保存路径
        
    Returns:
        str: 保存的文件路径
    """
    if filepath is None:
        ensure_dir(PATHS['model_dir'])
        filepath = PATHS['checkpoint'].format(generation)
    
    checkpoint_data = {
        'generation': generation,
        'population': [ind.get_weights() for ind in genetic_algorithm.population],
        'fitness_scores': genetic_algorithm.fitness_scores,
        'best_fitness': genetic_algorithm.best_fitness,
        'best_individual': genetic_algorithm.best_individual.get_weights() if genetic_algorithm.best_individual else None,
        'best_fitness_history': genetic_algorithm.best_fitness_history,
        'avg_fitness_history': genetic_algorithm.avg_fitness_history,
        'config': genetic_algorithm.config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"检查点已保存: {filepath}")
    
    return filepath


def load_checkpoint(filepath):
    """
    加载训练检查点
    
    Args:
        filepath: 检查点文件路径
        
    Returns:
        dict: 检查点数据
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")
    
    with open(filepath, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    print(f"检查点已加载: {filepath}")
    print(f"  代数: {checkpoint_data['generation']}")
    print(f"  最佳适应度: {checkpoint_data['best_fitness']:.2f}")
    
    return checkpoint_data


def save_training_log(log_data, filepath=None):
    """
    保存训练日志
    
    Args:
        log_data: 日志数据（字典或列表）
        filepath: 保存路径
    """
    if filepath is None:
        ensure_dir(PATHS['log_dir'])
        filepath = PATHS['log_file']
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"训练日志已保存: {filepath}")


def list_saved_models():
    """
    列出所有保存的模型
    
    Returns:
        list: 模型文件路径列表
    """
    model_dir = PATHS['model_dir']
    
    if not os.path.exists(model_dir):
        return []
    
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(model_dir, filename)
            models.append(filepath)
    
    return sorted(models)


def get_model_info(filepath):
    """
    获取模型信息（不加载权重）
    
    Args:
        filepath: 模型文件路径
        
    Returns:
        dict: 模型信息
    """
    info_filepath = filepath.replace('.pkl', '_info.json')
    
    if os.path.exists(info_filepath):
        with open(info_filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 如果没有info文件，加载pkl文件获取信息
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    return {
        'input_size': model_data['input_size'],
        'hidden_size': model_data['hidden_size'],
        'output_size': model_data['output_size'],
        'timestamp': model_data.get('timestamp', 'Unknown'),
        'weights_count': len(model_data['weights']),
        'metadata': model_data.get('metadata', {})
    }
