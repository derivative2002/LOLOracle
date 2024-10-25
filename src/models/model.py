import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SimpleClassifier(nn.Layer):
    """简单的分类模型"""

    def __init__(self, input_size, hidden_size, output_size):
        """初始化
        
        Args:
            input_size (int): 输入特征尺寸
            hidden_size (int): 隐藏层尺寸
            output_size (int): 输出尺寸
        """
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入张量

        Returns:
            Tensor: 输出张量
        """
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), axis=1)
        return x

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