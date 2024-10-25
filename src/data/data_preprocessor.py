import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理器"""

    def __init__(self):
        """初始化"""
        self.label_encoder = LabelEncoder()

    def preprocess(self, data, is_train=True):
        """数据预处理
        
        Args:
            data (pd.DataFrame): 原始数据
            is_train (bool): 是否为训练数据

        Returns:
            pd.DataFrame: 预处理后的数据
        """
        logger.info("开始数据预处理...")
        # 填充缺失值
        data = data.fillna(0)

        if is_train:
            # 处理标签
            data['win'] = self.label_encoder.fit_transform(data['win'])
        else:
            data['win'] = 0  # 占位

        # 特征列表，去掉不需要的列
        features = [col for col in data.columns if col not in ['id', 'win']]

        # 简单示例：归一化
        data[features] = data[features] / data[features].max()
        logger.info(f"预处理后数据形状：{data.shape}")
        logger.info("数据预处理完成。")
        return data