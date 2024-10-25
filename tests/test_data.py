import unittest
import pandas as pd
import yaml
from src.data.data_loader import DataLoader as MyDataLoader
from src.data.data_preprocessor import DataPreprocessor

class TestData(unittest.TestCase):
    """测试数据模块"""

    def test_data_loader(self):
        loader = MyDataLoader()
        train_data, test_data = loader.load_data()
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)

    def test_data_preprocessor(self):
        loader = MyDataLoader()
        train_data, _ = loader.load_data()
        # 加载配置文件
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        preprocessor = DataPreprocessor(config)
        X_train, X_val, y_train, y_val = preprocessor.preprocess(train_data, is_train=True)
        self.assertFalse(pd.isnull(X_train).any())
        self.assertFalse(pd.isnull(X_val).any())

if __name__ == '__main__':
    unittest.main()
