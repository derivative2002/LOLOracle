import sys
import os
import logging
import traceback
import paddle
import paddle.nn.functional as F
import paddle.distributed as dist  # 导入分布式训练模块
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import shutil

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置随机种子
from src.utils.utils import set_seed, get_device
set_seed(42)

# 读取配置文件
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 获取设备
device = get_device(config.get('device', 'cpu'))

# 获取模型名称和参数
model_name = config['model']['name']
model_params = config['model_params'][model_name]

# 检查是否使用分布式训练
use_distributed = config.get('distributed', False)

# 检查是否进行参数调优
tune_params = config.get('tune_params', False)

# 动态生成实验目录，防止覆盖
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_dir = f"outputs/experiments/{model_name}_{current_time}"
os.makedirs(experiment_dir, exist_ok=True)

# 日志配置
log_filename = os.path.join(experiment_dir, "train.log")
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 复制配置文件到实验目录
shutil.copy('config/config.yaml', os.path.join(experiment_dir, 'config.yaml'))

# 保存关键训练参数到文件
training_params_file = os.path.join(experiment_dir, 'training_parameters.txt')
with open(training_params_file, 'w') as f:
    f.write("关键训练参数：\n")
    f.write(f"模型名称: {model_name}\n")
    f.write(f"设备: {device}\n")
    f.write(f"分布式训练: {use_distributed}\n")
    f.write(f"训练参数: {model_params}\n")
    f.write(f"随机种子: 42\n")

# 数据加载与预处理
from src.data.data_loader import DataLoader as MyDataLoader
from src.data.data_preprocessor import DataPreprocessor

data_loader = MyDataLoader()
train_data, _ = data_loader.load_data()
preprocessor = DataPreprocessor(config)
X_train, X_val, y_train, y_val = preprocessor.preprocess(train_data, is_train=True)

logger.info(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
logger.info(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")

# Paddle 模型的训练流程
def train_paddle_model(model, model_save_path):
    # 获取 batch_size
    batch_size = model_params.get('batch_size', 64)

    # 创建数据集
    train_dataset = paddle.io.TensorDataset([
        paddle.to_tensor(X_train.values.astype('float32')),
        paddle.to_tensor(y_train.values.reshape(-1, 1).astype('float32'))
    ])
    val_dataset = paddle.io.TensorDataset([
        paddle.to_tensor(X_val.values.astype('float32')),
        paddle.to_tensor(y_val.values.reshape(-1, 1).astype('float32'))
    ])

    # 如果使用分布式训练，初始化并行环境
    if use_distributed:
        dist.init_parallel_env()
        logger.info("分布式训练环境已初始化")

    # 创建数据加载器
    num_workers = 4  # 您可以根据实际情况调整
    train_sampler = None
    if use_distributed:
        # 使用分布式采样器
        train_sampler = paddle.io.DistributedBatchSampler(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        train_loader = paddle.io.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            return_list=True)
    else:
        train_loader = paddle.io.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            return_list=True)

    # 将模型移动到设备
    model.to(device)

    # 如果使用分布式训练，封装模型
    if use_distributed:
        model = paddle.DataParallel(model)
        logger.info("模型已使用 DataParallel 封装")

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

        # 根据是否分布式调整进度条
        if not use_distributed or dist.get_rank() == 0:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        else:
            progress_bar = enumerate(train_loader)

        for batch_id, (x, y) in progress_bar:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = F.binary_cross_entropy_with_logits(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            batch_loss = loss.numpy().item()
            epoch_loss += batch_loss

            if not use_distributed or dist.get_rank() == 0:
                progress_bar.set_postfix(loss=batch_loss)

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        if not use_distributed or dist.get_rank() == 0:
            logger.info(f"Epoch [{epoch}] completed. Average Loss: {avg_epoch_loss:.4f}")

        # 验证模型
        model.eval()
        with paddle.no_grad():
            val_loader = paddle.io.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                return_list=True)
            val_loss_total = 0
            for x_val_batch, y_val_batch in val_loader:
                x_val_batch = x_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)
                y_val_pred = model(x_val_batch)
                val_loss = F.binary_cross_entropy_with_logits(y_val_pred, y_val_batch).numpy().item()
                val_loss_total += val_loss
            val_loss_avg = val_loss_total / len(val_loader)
            val_losses.append(val_loss_avg)
            if not use_distributed or dist.get_rank() == 0:
                logger.info(f"Validation Loss: {val_loss_avg:.4f}")

    # 保存模型（仅在主进程中）
    if not use_distributed or dist.get_rank() == 0:
        from src.models.model import save_paddle_model
        save_paddle_model(model, model_save_path)
        logger.info(f"模型已保存至 {model_save_path}")

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

        # 保存图表到实验目录
        figure_save_path = os.path.join(experiment_dir, f"{model_name}_training_curves_{current_time}.png")
        plt.savefig(figure_save_path)
        plt.close()
        logger.info(f"训练曲线已保存至 {figure_save_path}")

        # 在实验目录中保存训练损失和验证损失
        losses_save_path = os.path.join(experiment_dir, "losses.npz")
        np.savez(losses_save_path, train_losses=train_losses, val_losses=val_losses)
        logger.info(f"损失已保存至 {losses_save_path}")

# 主程序
try:
    from src.models.model import (
        LinearRegressionModel,
        MLPModel,
        FullyConnectedModel,
        XGBoostModel,
        LightGBMModel
    )

    input_size = X_train.shape[1]
    model_save_path = os.path.join(experiment_dir, f"{model_name}_model_{current_time}")

    if model_name == 'LinearRegression':
        model = LinearRegressionModel(input_size)
        model_save_path += '.pdparams'
        train_paddle_model(model, model_save_path)
    elif model_name == 'MLP':
        hidden_sizes = model_params.get('hidden_sizes', [128, 64])
        model = MLPModel(input_size, hidden_sizes)
        model_save_path += '.pdparams'
        train_paddle_model(model, model_save_path)
    elif model_name == 'FullyConnected':
        hidden_sizes = model_params.get('hidden_sizes', [256, 128, 64])
        model = FullyConnectedModel(input_size, hidden_sizes)
        model_save_path += '.pdparams'
        train_paddle_model(model, model_save_path)
    elif model_name == 'XGBoost':
        model = XGBoostModel(model_params)
        if tune_params:
            # 执行参数调优
            best_params = model.tune_parameters(X_train, y_train)
            # 更新模型参数为最佳参数
            model_params.update(best_params)
            # 重新初始化模型
            model = XGBoostModel(model_params)
        # 训练模型，传入验证集以查看损失
        model.train(X_train, y_train, X_val, y_val)
        model_save_path += '.joblib'
        model.save(model_save_path)
    elif model_name == 'LightGBM':
        model = LightGBMModel(model_params)
        model.train(X_train, y_train)
        model_save_path += '.joblib'
        model.save(model_save_path)
    else:
        logger.error(f'未支持的模型类型：{model_name}')
        sys.exit(1)

except Exception as e:
    logger.error("训练过程中出现错误：")
    logger.error(traceback.format_exc())
