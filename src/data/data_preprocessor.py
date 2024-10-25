import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class DataPreprocessor:
    def __init__(self, config):
        self.train_ratio = config['data_split']['train_ratio']
        self.random_state = config['data_split'].get('random_state', 42)
        self.scaler = StandardScaler()

    def preprocess(self, data, is_train=True):
        # 如果传入的是文件路径，则加载数据
        if isinstance(data, str):
            data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            data = data.copy()
        else:
            raise ValueError("数据格式不正确，应为 DataFrame 或文件路径。")

        # 数据清洗，处理缺失值
        data = data.dropna()

        # 特征与标签分离
        X = data.drop(['id', 'win'], axis=1, errors='ignore')
        if 'win' in data.columns:
            y = data['win']
        else:
            y = None

        # 特征缩放
        if is_train and y is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=self.train_ratio, random_state=self.random_state)

            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)

            return X_train, X_val, y_train.values.astype('float32'), y_val.values.astype('float32')
        else:
            X = self.scaler.transform(X)
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
