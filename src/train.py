import sys
import os
import logging
import traceback
import paddle
import paddle.nn.functional as F
import yaml
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# 日志配置
os.makedirs('logs', exist_ok=True)

# 动态生成日志文件名，防止覆盖
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"logs/train_{current_time}.log"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src.data.data_loader import DataLoader as MyDataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.model import LinearRegressionModel, MLPModel, FullyConnectedModel, save_model
from src.utils.utils import set_seed, get_device

def train():
    """训练模型"""
    # 设置随机种子
    set_seed(42)

    # 读取配置文件
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 获取设备
    device = get_device(config.get('device', 'cpu'))

    # 数据加载与预处理
    data_loader = MyDataLoader()
    train_data, _ = data_loader.load_data()
    preprocessor = DataPreprocessor(config)  # 传入 config 参数
    X_train, X_val, y_train, y_val = preprocessor.preprocess(train_data, is_train=True)

    logger.info(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    logger.info(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")

    # 获取模型名称和参数
    model_name = config['model']['name']
    model_params = config['model_params'][model_name]

    # 将模型参数记录到日志
    logger.info(f"Model parameters: {model_params}")

    # 获取 batch_size
    batch_size = model_params.get('batch_size', 64)  # 如果未设置，默认64

    # 创建数据集和数据加载器
    train_dataset = paddle.io.TensorDataset([paddle.to_tensor(X_train), paddle.to_tensor(y_train.reshape(-1, 1))])
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型定义
    input_size = X_train.shape[1]
    if model_name == 'LinearRegression':
        model = LinearRegressionModel(input_size)
    elif model_name == 'MLP':
        hidden_sizes = model_params.get('hidden_sizes', [128, 64])
        model = MLPModel(input_size, hidden_sizes)
    elif model_name == 'FullyConnected':
        hidden_sizes = model_params.get('hidden_sizes', [256, 128, 64])
        model = FullyConnectedModel(input_size, hidden_sizes)
    else:
        logger.error(f'未支持的模型类型：{model_name}')
        return

    model.to(device)

    # 优化器
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=model_params['learning_rate'])

    # 用于保存损失
    train_losses = []
    val_losses = []

    # 训练循环
    epochs = model_params['epochs']
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader()), total=len(train_loader()), desc=f"Epoch {epoch}/{epochs}")
        for batch_id, (x, y) in progress_bar:
            x = x.astype('float32')
            y = y.astype('float32')
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            batch_loss = loss.numpy().item()
            epoch_loss += batch_loss

            progress_bar.set_postfix(loss=batch_loss)

        avg_epoch_loss = epoch_loss / len(train_loader())
        train_losses.append(avg_epoch_loss)
        logger.info(f"Epoch [{epoch}] completed. Average Loss: {avg_epoch_loss:.4f}")

        # 验证模型
        model.eval()
        with paddle.no_grad():
            X_val_tensor = paddle.to_tensor(X_val.astype('float32')).to(device)
            y_val_tensor = paddle.to_tensor(y_val.reshape(-1, 1).astype('float32')).to(device)
            y_val_pred = model(X_val_tensor)
            val_loss = F.mse_loss(y_val_pred, y_val_tensor).numpy().item()
            val_losses.append(val_loss)
            logger.info(f"Validation Loss: {val_loss:.4f}")

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    model_save_path = f'outputs/models/{model_name}_model_{current_time}.pdparams'
    save_model(model, model_save_path)

    # 绘制损失曲线
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    os.makedirs('outputs/figures', exist_ok=True)
    figure_save_path = f'outputs/figures/{model_name}_training_curves_{current_time}.png'
    plt.savefig(figure_save_path)
    plt.close()
    logger.info(f"训练曲线已保存至 {figure_save_path}")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        logger.error(f"训练过程中出现异常：{e}")
        traceback.print_exc()
