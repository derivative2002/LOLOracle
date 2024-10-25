import sys
import os
import paddle
import paddle.nn.functional as F
import pandas as pd
import numpy as np
import logging
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import DataLoader as MyDataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.model import (
    LinearRegressionModel,
    MLPModel,
    FullyConnectedModel,
    XGBoostModel,
    LightGBMModel,
    load_paddle_model
)
from src.utils.utils import get_device

logger = logging.getLogger(__name__)

def predict():
    """模型预测"""
    # 日志配置
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
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
    preprocessor = DataPreprocessor(config)
    X_test = preprocessor.preprocess(test_data, is_train=False)

    # 模型定义并加载
    model_name = config['model']['name']
    input_size = X_test.shape[1]
    experiments_dir = 'outputs/experiments'
    model = None
    model_save_path = ''

    # 查找最新的实验目录和模型文件
    experiment_dirs = [os.path.join(experiments_dir, d) for d in os.listdir(experiments_dir)
                       if os.path.isdir(os.path.join(experiments_dir, d)) and d.startswith(model_name)]
    if not experiment_dirs:
        logger.error("未找到实验目录，请先训练模型。")
        return
    experiment_dirs.sort()
    latest_experiment_dir = experiment_dirs[-1]
    model_files = [f for f in os.listdir(latest_experiment_dir)
                   if f.endswith('.pdparams') or f.endswith('.joblib')]
    if not model_files:
        logger.error("未找到模型参数文件。")
        return
    model_files.sort()
    model_save_path = os.path.join(latest_experiment_dir, model_files[-1])

    if model_name == 'LinearRegression':
        model = LinearRegressionModel(input_size)
        model.to(device)
        load_paddle_model(model, model_save_path)
    elif model_name == 'MLP':
        hidden_sizes = config['model_params'][model_name].get('hidden_sizes', [128, 64])
        model = MLPModel(input_size, hidden_sizes)
        model.to(device)
        load_paddle_model(model, model_save_path)
    elif model_name == 'FullyConnected':
        hidden_sizes = config['model_params'][model_name].get('hidden_sizes', [256, 128, 64])
        model = FullyConnectedModel(input_size, hidden_sizes)
        model.to(device)
        load_paddle_model(model, model_save_path)
    elif model_name == 'XGBoost':
        model = XGBoostModel(config['model_params'][model_name])
        model.load(model_save_path)
    elif model_name == 'LightGBM':
        model = LightGBMModel(config['model_params'][model_name])
        model.load(model_save_path)
    else:
        logger.error(f'未支持的模型类型：{model_name}')
        return

    # 进行预测
    if model_name in ['LinearRegression', 'MLP', 'FullyConnected']:
        model.eval()
        with paddle.no_grad():
            X_test_tensor = paddle.to_tensor(X_test.values.astype('float32')).to(device)
            outputs = model(X_test_tensor)
            predictions = (F.sigmoid(outputs) >= 0.5).astype('int64').numpy().flatten()
    elif model_name in ['XGBoost', 'LightGBM']:
        predictions = model.predict(X_test)
    else:
        logger.error(f'未支持的模型类型：{model_name}')
        return

    # 保存预测结果到对应的实验目录
    submission = pd.DataFrame({'win': predictions})
    prediction_save_path = os.path.join(latest_experiment_dir, 'submission.csv')
    submission.to_csv(prediction_save_path, index=False)
    logger.info(f"预测结果已保存至 {prediction_save_path}")

if __name__ == '__main__':
    predict()
