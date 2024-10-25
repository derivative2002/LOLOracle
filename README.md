# 英雄联盟大师预测项目

## 项目简介

本项目旨在参加百度飞桨（PaddlePaddle）学习赛：**英雄联盟大师预测**赛题，构建一个分类模型，根据英雄联盟玩家的实时游戏数据，预测玩家在本局游戏中的输赢情况。

## 赛题背景

英雄联盟是一款深受全球玩家喜爱的多人在线战术竞技游戏。本次比赛以英雄联盟手游为背景，提供了玩家的游戏对局数据，包括击杀数、伤害量等多种特征。我们的目标是从这些数据中挖掘出影响游戏结果的关键因素，构建模型预测玩家的输赢。

### 比赛任务

本赛题数据为英雄联盟玩家的实时游戏数据，记录了用户在游戏中的对局数据，如击杀数、物理伤害等。希望参赛选手能从数据集中挖掘出数据的规律，并预测玩家在本局游戏中的输赢情况。

### 数据集介绍

- **训练集**：共800万条数据
- **测试集**：共2万条数据

### 数据说明

数据集中每一行为一个玩家的游戏数据，数据字段如下所示：

- `id`：玩家记录ID
- `win`：是否胜利，标签变量
- `kills`：击杀次数
- `deaths`：死亡次数
- `assists`：助攻次数
- `largestkillingspree`：最大击杀连续数（Killing Spree）
- `largestmultikill`：最大多重击杀（Multi Kill）
- `longesttimespentliving`：最长存活时间
- `doublekills`：双杀次数
- `triplekills`：三杀次数
- `quadrakills`：四杀次数
- `pentakills`：五杀次数
- `totdmgdealt`：总伤害
- `magicdmgdealt`：魔法伤害
- `physicaldmgdealt`：物理伤害
- `truedmgdealt`：真实伤害
- `largestcrit`：最大暴击伤害
- `totdmgtochamp`：对敌方英雄的总伤害
- `magicdmgtochamp`：对敌方英雄的魔法伤害
- `physdmgtochamp`：对敌方英雄的物理伤害
- `truedmgtochamp`：对敌方英雄的真实伤害
- `totheal`：治疗量
- `totunitshealed`：被治愈的单位总数
- `dmgtoturrets`：对防御塔的伤害
- `timecc`：控制时间
- `totdmgtaken`：承受的总伤害
- `magicdmgtaken`：承受的魔法伤害
- `physdmgtaken`：承受的物理伤害
- `truedmgtaken`：承受的真实伤害
- `wardsplaced`：放置侦查守卫次数
- `wardskilled`：摧毁侦查守卫次数
- `firstblood`：是否为一血

测试集中标签字段 `win` 为空，需要选手进行预测。

### 评测指标

使用 **准确率（Accuracy）** 作为评测指标：

```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
score = accuracy_score(y_true, y_pred)
print(f"准确率：{score}")
```

## 提交结果要求

- **提交内容及格式**

  1. 本次比赛要求参赛选手**必须使用飞桨（PaddlePaddle）深度学习框架训练的模型**。
  2. 结果文件命名：`submission.zip`。
  3. 结果文件格式：`zip` 文件格式，`zip` 文件解压后为一个 `submission.csv` 文件，编码为 `UTF-8`。
  4. 结果文件内容：`submission.csv` 仅包含一个字段，为 `win` 字段。

- **提交示例**

  ```csv
  win
  0
  1
  1
  ...
  0
  ```

- **提交注意事项**

  1. 每支队伍每天参与评测的提交次数不超过 5 次，排行榜将按照评测分数从高到低排序，并实时更新。
  2. 排行榜中只显示每支队伍历史提交结果的最高成绩，各支队伍可在提交结果页面的个人成绩中查看历史提交记录。

## 项目结构

```
.
├── config                      # 配置文件目录
│   ├── config.yaml             # 配置文件
│   └── logging.conf            # 日志配置文件
├── data                        # 数据目录
│   ├── processed               # 预处理后的数据
│   └── raw                     # 原始数据
├── logs                        # 日志文件目录
├── notebooks                   # Notebook目录
│   └── EDA.ipynb               # 数据探索性分析
├── outputs                     # 输出结果目录
│   ├── experiments             # 实验结果
│   ├── models                  # 保存的模型
│   └── predictions             # 预测结果
├── README.md                   # 项目说明文件
├── requirements.txt            # 依赖库列表
├── src                         # 源代码目录
│   ├── data                    # 数据处理模块
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   │   ├── preprocess.py
│   │   └── postprocess.py
│   ├── models                  # 模型构建模块
│   │   ├── __init__.py
│   │   └── model.py
│   ├── predict.py              # 预测脚本
│   ├── train.py                # 训练脚本
│   └── utils                   # 工具函数模块
│       ├── __init__.py
│       └── utils.py
└── tests                       # 测试模块
    ├── __init__.py
    ├── test_data.py
    ├── test_models.py
    └── test_utils.py
```

## 环境依赖

请确保您的环境满足以下条件：

- **Python 3.7** 或以上版本
- 已安装 **PaddlePaddle** 深度学习框架

您可以使用以下命令安装项目所需的依赖库：

```bash
pip install -r requirements.txt
```

`requirements.txt` 文件内容：

```
paddlepaddle
pandas
numpy
scikit-learn
PyYAML
matplotlib
tqdm
```

## 数据准备

请将比赛提供的原始数据文件放置于 `data/raw/` 目录下：

- 训练集：`train.csv`
- 测试集：`test.csv`

## 配置说明

项目的配置文件位于 `config/config.yaml`，您可以根据需要修改其中的参数：

```yaml
# 配置文件

# 数据路径配置
data:
  train_data_path: 'data/raw/train.csv'
  test_data_path: 'data/raw/test.csv'
  processed_train_data: 'data/processed/processed_train.csv'
  processed_test_data: 'data/processed/processed_test.csv'

# 数据集划分比例
data_split:
  train_ratio: 0.8         # 训练集比例，可根据需要调整
  random_state: 42         # 随机种子，可根据需要调整

# 设备配置
device: 'cpu'              # 可选 'cpu' 或 'gpu'

# 是否使用分布式训练
distributed: false         # 如果需要分布式训练，设置为 true

# 是否进行参数调优
tune_params: true          # 如果需要进行参数调优，设置为 true

# 模型选择
model:
  name: 'MLP'              # 可选模型：'LinearRegression'、'MLP'、'FullyConnected'、'XGBoost'、'LightGBM'

# 模型参数配置
model_params:
  LinearRegression:
    learning_rate: 0.001
    epochs: 30
    batch_size: 128

  MLP:
    epochs: 30
    learning_rate: 0.001
    batch_size: 128
    hidden_sizes: [256, 128, 64, 32]

  FullyConnected:
    hidden_sizes: [256, 128, 64]

# 日志配置文件路径
logging_config: 'config/logging.conf'
```

## 日志配置

日志配置文件位于 `config/logging.conf`，您可以根据需要调整日志级别和格式。默认情况下，日志将同时输出到控制台和日志文件。

## 运行步骤

### 1. 训练模型

在命令行中运行以下命令开始训练：

```bash
python src/train.py
```

训练过程中，模型将根据配置文件中的参数进行训练。训练完成后，模型参数将保存在 `outputs/experiments/` 目录下。

### 2. 模型预测

使用训练好的模型对测试集进行预测：

```bash
python src/predict.py
```

预测结果将保存至 `outputs/predictions/submission.csv`，文件格式符合比赛要求，可直接用于提交。

### 3. 结果提交

根据比赛要求，将 `submission.csv` 文件压缩为 `submission.zip`，然后提交至比赛平台。

## 代码说明

### 训练脚本 `src/train.py`

```python
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
    f.write(f"Model Name: {model_name}\n")
    f.write(f"Model Parameters: {model_params}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Use Distributed: {use_distributed}\n")
    f.write(f"Tune Parameters: {tune_params}\n")

# 导入数据处理和模型相关模块
from src.data.data_loader import DataLoader as MyDataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.model import (
    LinearRegressionModel,
    MLPModel,
    FullyConnectedModel,
    XGBoostModel,
    LightGBMModel,
    save_paddle_model,
    save_sklearn_model
)

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
            X_val_tensor = paddle.to_tensor(X_val.values.astype('float32')).to(device)
            y_val_tensor = paddle.to_tensor(y_val.values.reshape(-1, 1).astype('float32')).to(device)
            y_val_pred = model(X_val_tensor)
            val_loss = F.mse_loss(y_val_pred, y_val_tensor).numpy().item()
            val_losses.append(val_loss)
            logger.info(f"Validation Loss: {val_loss:.4f}")

    # 保存模型
    model_save_path = os.path.join(experiment_dir, f"{model_name}_model.pdparams")
    save_paddle_model(model, model_save_path)

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
    figure_save_path = os.path.join(experiment_dir, f"{model_name}_training_curves.png")
    plt.savefig(figure_save_path)
    plt.close()
    logger.info(f"训练曲线已保存至 {figure_save_path}")

# 根据模型名称选择训练流程
if model_name in ['LinearRegression', 'MLP', 'FullyConnected']:
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
        exit(1)

    train_paddle_model(model, experiment_dir)

else:
    logger.error(f'未支持的模型类型：{model_name}')
    exit(1)
```

### 预测脚本 `src/predict.py`

```python
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

    # 找到最新的实验目录和模型文件
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
    else:
        logger.error(f'未支持的模型类型：{model_name}')
        return

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
```

## 数据探索

在 `notebooks/EDA.ipynb` 中，可以进行数据的探索性分析。您可以根据需要添加更多的分析内容，以加深对数据的理解。

## 项目特点

- **模块化设计**：代码按照功能模块划分，清晰明确，方便维护和扩展。
- **配置化管理**：使用 YAML 配置文件集中管理参数，便于实验调参。
- **日志功能完善**：详细的日志记录，方便跟踪程序运行和调试。
- **单元测试**：为关键模块编写了测试用例，保证代码的稳定性和可靠性。
- **实验记录完备**：每次训练都会生成独立的实验目录，保存模型、配置和日志，方便比较和回溯。

## 注意事项

- 请确保使用的 PaddlePaddle 版本符合比赛要求。
- 运行训练和预测脚本前，请检查数据路径和配置文件是否正确。
- 如需修改模型结构或参数，请在 `config/config.yaml` 中进行调整。
- 如果需要使用 GPU，加速模型训练，请在配置文件中将 `device` 设置为 `'gpu'`。

## 参考资料

- [百度飞桨（PaddlePaddle）官网](https://www.paddlepaddle.org.cn/)
- [比赛链接](https://aistudio.baidu.com/competition/detail/797/0/leaderboard)

---

希望本项目能够帮助您在比赛中取得优异的成绩！