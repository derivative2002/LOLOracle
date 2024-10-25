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
from src.models.model import LinearRegressionModel, MLPModel, FullyConnectedModel, load_model
from src.utils.utils import get_device

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

    # 获取设备
    device = get_device(config.get('device', 'cpu'))

    # 数据加载与预处理
    data_loader = MyDataLoader()
    _, test_data = data_loader.load_data()
    preprocessor = DataPreprocessor(config)  # 传入 config 参数
    X_test = preprocessor.preprocess(test_data, is_train=False)

    # 模型定义并加载
    model_name = config['model']['name']
    input_size = X_test.shape[1]
    if model_name == 'LinearRegression':
        model = LinearRegressionModel(input_size)
    elif model_name == 'MLP':
        hidden_sizes = config['model_params'][model_name].get('hidden_sizes', [128, 64])
        model = MLPModel(input_size, hidden_sizes)
    elif model_name == 'FullyConnected':
        hidden_sizes = config['model_params'][model_name].get('hidden_sizes', [256, 128, 64])
        model = FullyConnectedModel(input_size, hidden_sizes)
    else:
        logger.error(f'未支持的模型类型：{model_name}')
        return

    model.to(device)

    # 找到最新的模型文件
    models_dir = 'outputs/models'
    model_files = [f for f in os.listdir(models_dir) if f.startswith(f'{model_name}_model')]
    if not model_files:
        logger.error("未找到模型文件，请先训练模型。")
        return
    # 选择最新的模型文件
    model_files.sort()
    latest_model_file = model_files[-1]
    model_path = os.path.join(models_dir, latest_model_file)

    # 加载模型参数
    load_model(model, model_path)

    # 进行预测
    model.eval()
    with paddle.no_grad():
        X_test_tensor = paddle.to_tensor(X_test.astype('float32')).to(device)
        outputs = model(X_test_tensor)
        predictions = outputs.numpy().flatten()

    # 根据回归结果生成分类结果（根据阈值，比如0.5）
    predictions = (predictions >= 0.5).astype(int)

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
