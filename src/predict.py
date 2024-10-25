import paddle
import pandas as pd
import numpy as np
import logging
import yaml

from src.data.data_loader import DataLoader as MyDataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.model import SimpleClassifier, load_model

logger = logging.getLogger(__name__)

def predict():
    """模型预测"""
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
    model = SimpleClassifier(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        output_size=config['model']['output_size']
    )
    load_model(model, 'outputs/models/model.pdparams')

    # 进行预测
    model.eval()
    with paddle.no_grad():
        outputs = model(paddle.to_tensor(X_test))
        predictions = paddle.argmax(outputs, axis=1).numpy()

    # 保存预测结果
    submission = pd.DataFrame({'win': predictions})
    submission.to_csv('outputs/predictions/submission.csv', index=False)
    logger.info("预测结果已保存至 outputs/predictions/submission.csv")

if __name__ == '__main__':
    predict()

