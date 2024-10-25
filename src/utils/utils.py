import paddle
import numpy as np
import random

def set_seed(seed=42):
    """设置随机种子
    
    Args:
        seed (int): 随机种子
    """
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

