import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class AdvancedClassifier(nn.Layer):
    """改进的分类模型"""

    def __init__(self, input_size, hidden_sizes, output_size):
        """初始化

        Args:
            input_size (int): 输入特征尺寸
            hidden_sizes (list): 隐藏层尺寸列表
            output_size (int): 输出尺寸
        """
        super(AdvancedClassifier, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播

        Args:
            x (Tensor): 输入张量

        Returns:
            Tensor: 输出张量
        """
        return self.network(x)

def save_model(model, path):
    """保存模型
    
    Args:
        model (nn.Layer): 模型
        path (str): 保存路径
    """
    paddle.save(model.state_dict(), path)
    logger.info(f"模型已保存至 {path}")

def load_model(model, path):
    """加载模型
    
    Args:
        model (nn.Layer): 模型
        path (str): 模型路径
    """
    model.set_state_dict(paddle.load(path))
    logger.info(f"模型已从 {path} 加载")
