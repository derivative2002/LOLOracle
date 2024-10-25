import pandas as pd
import logging
import yaml

logger = logging.getLogger(__name__)

class DataLoader:
    """数据加载器"""

    def __init__(self, config_path='config/config.yaml'):
        """初始化
        
        Args:
            config_path (str): 配置文件路径
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.train_data_path = self.config['data']['train_data_path']
        self.test_data_path = self.config['data']['test_data_path']

    def load_data(self):
        """加载数据
        
        Returns:
            pd.DataFrame: 训练数据
            pd.DataFrame: 测试数据
        """
        logger.info("开始加载数据...")
        train_data = pd.read_csv(self.train_data_path)
        test_data = pd.read_csv(self.test_data_path)
        logger.info("数据加载完成。")
        return train_data, test_data

