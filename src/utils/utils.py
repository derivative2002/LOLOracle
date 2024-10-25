import paddle
import numpy as np
import random

def set_seed(seed=42):
    """设置随机种子"""
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_device(device_string):
    """获取设备"""
    if device_string.lower() == 'gpu' and paddle.is_compiled_with_cuda():
        return paddle.CUDAPlace(0)
    else:
        return paddle.CPUPlace()
