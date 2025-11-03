"""
神经网络类 - 机器人的"大脑"

这个神经网络的权重就是遗传算法要优化的"基因"

工作原理：
1. 输入层：接收传感器数据（8个距离传感器 + 3个地面传感器）
2. 隐藏层：处理和组合传感器信息
3. 输出层：产生控制指令（左轮速度、右轮速度）

神经网络的所有权重和偏置就是"基因"，遗传算法会不断优化这些参数
"""
import numpy as np


class NeuralNetwork:
    """
    简单的前馈神经网络

    这个网络将传感器输入映射到电机输出
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化神经网络

        Args:
            input_size: 输入层大小（传感器数量）
            hidden_size: 隐藏层大小
            output_size: 输出层大小（电机数量）
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 随机初始化权重和偏置（这些就是"基因"）
        # 使用较小的初始值避免饱和
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
        self.bias_hidden = np.random.randn(hidden_size) * 0.5

        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
        self.bias_output = np.random.randn(output_size) * 0.5

    def forward(self, inputs):
        """
        前向传播 - 根据传感器输入计算轮速输出

        这是神经网络的"思考"过程：
        传感器数据 -> 隐藏层处理 -> 输出控制指令

        Args:
            inputs: 传感器数据数组 [11个值]

        Returns:
            outputs: [左轮速度, 右轮速度]，范围[-1, 1]
        """
        # 第一层：输入层 -> 隐藏层
        hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden = np.tanh(hidden)  # tanh激活函数，输出范围[-1, 1]

        # 第二层：隐藏层 -> 输出层
        outputs = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        outputs = np.tanh(outputs)  # tanh激活函数，输出范围[-1, 1]

        return outputs

    def get_weights(self):
        """
        获取所有权重（基因）

        将所有权重和偏置展平成一维数组，这就是"基因组"
        遗传算法会操作这个数组来进化机器人

        Returns:
            一维numpy数组，包含所有权重和偏置
        """
        return np.concatenate([
            self.weights_input_hidden.flatten(),
            self.bias_hidden.flatten(),
            self.weights_hidden_output.flatten(),
            self.bias_output.flatten()
        ])

    def set_weights(self, weights):
        """
        设置所有权重（基因）

        从一维数组恢复所有权重和偏置
        这用于将父代的基因传给子代

        Args:
            weights: 一维numpy数组，包含所有权重和偏置
        """
        idx = 0

        # 恢复输入层->隐藏层的权重
        size = self.input_size * self.hidden_size
        self.weights_input_hidden = weights[idx:idx+size].reshape(
            self.input_size, self.hidden_size
        )
        idx += size

        # 恢复隐藏层偏置
        self.bias_hidden = weights[idx:idx+self.hidden_size]
        idx += self.hidden_size

        # 恢复隐藏层->输出层的权重
        size = self.hidden_size * self.output_size
        self.weights_hidden_output = weights[idx:idx+size].reshape(
            self.hidden_size, self.output_size
        )
        idx += size

        # 恢复输出层偏置
        self.bias_output = weights[idx:idx+self.output_size]

    def get_weights_count(self):
        """
        获取权重总数（基因长度）

        Returns:
            权重总数
        """
        return (self.input_size * self.hidden_size +
                self.hidden_size +
                self.hidden_size * self.output_size +
                self.output_size)

    def copy(self):
        """
        复制神经网络

        创建一个完全相同的副本

        Returns:
            新的神经网络对象
        """
        new_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_nn.set_weights(self.get_weights().copy())
        return new_nn
"""
神经网络类 - 机器人的"大脑"

这个神经网络的权重就是遗传算法要优化的"基因"

工作原理：
1. 输入层：接收传感器数据（8个距离传感器 + 3个地面传感器）
2. 隐藏层：处理和组合传感器信息
3. 输出层：产生控制指令（左轮速度、右轮速度）

神经网络的所有权重和偏置就是"基因"，遗传算法会不断优化这些参数
"""
import numpy as np


class NeuralNetwork:
    """
    简单的前馈神经网络
    
    这个网络将传感器输入映射到电机输出
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化神经网络
        
        Args:
            input_size: 输入层大小（传感器数量）
            hidden_size: 隐藏层大小
            output_size: 输出层大小（电机数量）
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 随机初始化权重和偏置（这些就是"基因"）
        # 使用较小的初始值避免饱和
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
        self.bias_hidden = np.random.randn(hidden_size) * 0.5
        
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
        self.bias_output = np.random.randn(output_size) * 0.5
    
    def forward(self, inputs):
        """
        前向传播 - 根据传感器输入计算轮速输出
        
        这是神经网络的"思考"过程：
        传感器数据 -> 隐藏层处理 -> 输出控制指令
        
        Args:
            inputs: 传感器数据数组 [11个值]
            
        Returns:
            outputs: [左轮速度, 右轮速度]，范围[-1, 1]
        """
        # 第一层：输入层 -> 隐藏层
        hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden = np.tanh(hidden)  # tanh激活函数，输出范围[-1, 1]
        
        # 第二层：隐藏层 -> 输出层
        outputs = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        outputs = np.tanh(outputs)  # tanh激活函数，输出范围[-1, 1]
        
        return outputs
    
    def get_weights(self):
        """
        获取所有权重（基因）
        
        将所有权重和偏置展平成一维数组，这就是"基因组"
        遗传算法会操作这个数组来进化机器人
        
        Returns:
            一维numpy数组，包含所有权重和偏置
        """
        return np.concatenate([
            self.weights_input_hidden.flatten(),
            self.bias_hidden.flatten(),
            self.weights_hidden_output.flatten(),
            self.bias_output.flatten()
        ])
    
    def set_weights(self, weights):
        """
        设置所有权重（基因）
        
        从一维数组恢复所有权重和偏置
        这用于将父代的基因传给子代
        
        Args:
            weights: 一维numpy数组，包含所有权重和偏置
        """
        idx = 0
        
        # 恢复输入层->隐藏层的权重
        size = self.input_size * self.hidden_size
        self.weights_input_hidden = weights[idx:idx+size].reshape(
            self.input_size, self.hidden_size
        )
        idx += size
        
        # 恢复隐藏层偏置
        self.bias_hidden = weights[idx:idx+self.hidden_size]
        idx += self.hidden_size
        
        # 恢复隐藏层->输出层的权重
        size = self.hidden_size * self.output_size
        self.weights_hidden_output = weights[idx:idx+size].reshape(
            self.hidden_size, self.output_size
        )
        idx += size
        
        # 恢复输出层偏置
        self.bias_output = weights[idx:idx+self.output_size]
    
    def get_weights_count(self):
        """
        获取权重总数（基因长度）
        
        Returns:
            权重总数
        """
        return (self.input_size * self.hidden_size + 
                self.hidden_size + 
                self.hidden_size * self.output_size + 
                self.output_size)
    
    def copy(self):
        """
        复制神经网络
        
        创建一个完全相同的副本
        
        Returns:
            新的神经网络对象
        """
        new_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_nn.set_weights(self.get_weights().copy())
        return new_nn
