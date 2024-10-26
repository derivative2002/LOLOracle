import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import joblib  # 用于保存和加载 scaler

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class DataPreprocessor:
    def __init__(self, config, is_train=True):
        self.config = config
        self.random_state = config['data_split']['random_state']
        self.train_ratio = config['data_split']['train_ratio']
        self.is_train = is_train
        self.scaler = None

        if self.is_train:
            self.scaler = StandardScaler()
        else:
            # 在预测阶段，加载已保存的 scaler
            self.scaler = joblib.load('outputs/scaler.joblib')

    def preprocess(self, data):
        data = data.copy()

        # 数据清洗（可根据需要添加）

        # 提取特征和标签
        if 'win' in data.columns:
            X = data.drop(columns=['id', 'win'])
            y = data['win'].astype('float32')  # 确保标签为 float32 类型
        else:
            X = data.drop(columns=['id'])
            y = None

        # 特征标准化
        if self.is_train:
            X = self.scaler.fit_transform(X)
            # 保存已拟合的 scaler
            joblib.dump(self.scaler, 'outputs/scaler.joblib')
        else:
            X = self.scaler.transform(X)

        if self.is_train:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=self.train_ratio, random_state=self.random_state)
            return X_train, X_val, y_train, y_val
        else:
            return X

    def preprocess_test(self, test_data_path):
        # 加载测试数据
        test_data = pd.read_csv(test_data_path)

        # 数据清洗，处理缺失值
        test_data = test_data.dropna()

        # 特征提取
        X_test = test_data.drop(['id', 'win'], axis=1, errors='ignore')

        # 特征缩放
        X_test = self.scaler.transform(X_test)

        return X_test
