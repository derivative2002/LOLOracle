import paddle
import paddle.nn as nn
import logging

logger = logging.getLogger(__name__)

class LinearRegressionModel(nn.Layer):
    """线性回归模型"""
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

class MLPModel(nn.Layer):
    """多层感知机模型"""
    def __init__(self, input_size, hidden_sizes):
        super(MLPModel, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FullyConnectedModel(nn.Layer):
    """全连接神经网络模型"""
    def __init__(self, input_size, hidden_sizes):
        super(FullyConnectedModel, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def save_model(model, path):
    """保存模型"""
    paddle.save(model.state_dict(), path)
    logger.info(f"模型已保存至 {path}")

def load_model(model, path):
    """加载模型"""
    model.set_state_dict(paddle.load(path))
    logger.info(f"模型已从 {path} 加载")
