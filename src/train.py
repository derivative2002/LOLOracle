import sys
import os
import logging
import traceback

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader, TensorDataset
import yaml
import numpy as np

from src.data.data_loader import DataLoader as MyDataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.model import SimpleClassifier, save_model

def train():
    """训练模型"""
    # 读取配置文件
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 数据加载与预处理
    data_loader = MyDataLoader()
    train_data, _ = data_loader.load_data()
    preprocessor = DataPreprocessor()
    train_data = preprocessor.preprocess(train_data)

    # 采样部分数据进行测试
    train_data = train_data.sample(n=10000, random_state=42)

    # 准备训练数据
    features = [col for col in train_data.columns if col not in ['id', 'win']]
    X_train = train_data[features].values.astype('float32')
    y_train = train_data['win'].values.astype('int64')

    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")

    train_dataset = TensorDataset([X_train, y_train])
    train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=True)

    # 模型定义
    model = SimpleClassifier(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        output_size=config['model']['output_size']
    )

    # 优化器
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=config['model']['learning_rate'])

    # 训练循环
    for epoch in range(config['model']['epochs']):
        for batch_id, (x, y) in enumerate(train_loader()):
            out = model(x)
            loss = F.cross_entropy(out, y)
            avg_loss = paddle.mean(loss)
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if batch_id % 100 == 0:
                logger.info(f"Epoch [{epoch}/{config['model']['epochs']}], Step [{batch_id}], Loss: {avg_loss.numpy()[0]:.4f}")

    # 保存模型
    save_model(model, 'outputs/models/model.pdparams')

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        logger.error(f"训练过程中出现异常：{e}")
        traceback.print_exc()
