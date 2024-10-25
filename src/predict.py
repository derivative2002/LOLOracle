import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import paddle
import pandas as pd
import numpy as np
import logging
import yaml

from src.data.data_loader import DataLoader as MyDataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.model import AdvancedClassifier, load_model

logger = logging.getLogger(__name__)

def predict():
    """模型预测"""
    # 日志配置
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("logs/predict.log", mode='w', encoding='utf-8'),
                            logging.StreamHandler()
                        ])

    # 读取配置文件
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 数据加载与预处理
    data_loader = MyDataLoader()
    _, test_data = data_loader.load_data()
    preprocessor = DataPreprocessor()
    test_data = preprocessor.preprocess(test_data, is_train=False)

    # 准备测试数据
    features = [col for col in test_data.columns if col not in ['id', 'win']]
    X_test = test_data[features].values.astype('float32')

    # 模型定义并加载
    model = AdvancedClassifier(
        input_size=config['model']['input_size'],
        hidden_sizes=config['model']['hidden_sizes'],
        output_size=config['model']['output_size']
    )
    load_model(model, 'outputs/models/model.pdparams')

    # 进行预测
    model.eval()
    with paddle.no_grad():
        outputs = model(paddle.to_tensor(X_test))
        predictions = paddle.argmax(outputs, axis=1).numpy()

    # 保存预测结果之前，检查并创建目录
    output_dir = 'outputs/predictions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存预测结果
    submission = pd.DataFrame({'win': predictions})
    submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
    logger.info("预测结果已保存至 outputs/predictions/submission.csv")

if __name__ == '__main__':
    predict()
