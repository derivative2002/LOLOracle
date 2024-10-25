import unittest
import pandas as pd
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
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(train_data)
        self.assertFalse(processed_data.isnull().values.any())

if __name__ == '__main__':
    unittest.main()

