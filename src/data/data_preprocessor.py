import pandas as pd
import logging
import pickle  # 用于保存和加载缩放器
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理器"""

    def __init__(self):
        """初始化"""
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def preprocess(self, data, is_train=True):
        """数据预处理

        Args:
            data (pd.DataFrame): 原始数据
            is_train (bool): 是否为训练数据

        Returns:
            pd.DataFrame: 预处理后的数据
        """
        logger.info("开始数据预处理...")
        data = data.fillna(0)

        # 特征列表，去掉不需要的列
        features = [col for col in data.columns if col not in ['id', 'win']]

        if is_train:
            # 编码标签
            data['win'] = self.label_encoder.fit_transform(data['win'])
            # 特征标准化
            data[features] = self.scaler.fit_transform(data[features])
            # 保存缩放器和编码器
            with open('outputs/models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            with open('outputs/models/label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
        else:
            # 加载缩放器和编码器
            with open('outputs/models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('outputs/models/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            # 特征标准化
            data[features] = self.scaler.transform(data[features])
            data['win'] = 0  # 占位

        logger.info(f"预处理后数据形状：{data.shape}")
        logger.info("数据预处理完成。")
        return data
