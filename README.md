
# 英雄联盟大师预测项目

## 项目简介

本项目旨在参加百度飞桨（PaddlePaddle）学习赛：**英雄联盟大师预测**赛题，构建一个分类模型，根据英雄联盟玩家的实时游戏数据，预测玩家在本局游戏中的输赢情况。

## 赛题背景

英雄联盟是一款深受全球玩家喜爱的多人在线战术竞技游戏。本次比赛以英雄联盟手游为背景，提供了玩家的游戏对局数据，包括击杀数、伤害量等多种特征。我们的目标是从这些数据中挖掘出影响游戏结果的关键因素，构建模型预测玩家的输赢。

## 项目结构

```
.
├── config
│   ├── config.yaml          # 配置文件
│   └── logging.conf         # 日志配置文件
├── data
│   ├── processed            # 预处理后的数据
│   └── raw                  # 原始数据
├── logs                     # 日志文件
├── notebooks
│   └── EDA.ipynb            # 数据探索性分析
├── outputs
│   ├── models               # 保存的模型
│   └── predictions          # 预测结果
├── README.md                # 项目说明文件
├── requirements.txt         # 依赖库列表
├── src
│   ├── data                 # 数据处理模块
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   │   ├── preprocess.py
│   │   └── postprocess.py
│   ├── models               # 模型构建模块
│   │   ├── __init__.py
│   │   └── model.py
│   ├── predict.py           # 预测脚本
│   ├── train.py             # 训练脚本
│   └── utils                # 工具函数模块
│       ├── __init__.py
│       └── utils.py
└── tests                    # 测试模块
    ├── __init__.py
    ├── test_data.py
    ├── test_models.py
    └── test_utils.py
```

## 环境依赖

请确保您的环境满足以下条件：

- Python 3.7 或以上版本
- 已安装 PaddlePaddle 深度学习框架

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
```

## 数据准备

请将比赛提供的原始数据文件放置于 `data/raw/` 目录下：

- 训练集：`train.csv`
- 测试集：`test.csv`

## 配置说明

项目的配置文件位于 `config/config.yaml`，您可以根据需要修改其中的参数：

```yaml
# config/config.yaml

# 数据路径配置
data:
  train_data_path: 'data/raw/train.csv'
  test_data_path: 'data/raw/test.csv'
  processed_train_data: 'data/processed/processed_train.csv'
  processed_test_data: 'data/processed/processed_test.csv'

# 模型参数配置
model:
  input_size: 24           # 输入特征数
  hidden_size: 64          # 隐藏层神经元数
  output_size: 2           # 输出类别数
  learning_rate: 0.001     # 学习率
  epochs: 10               # 训练轮数
  batch_size: 128          # 批次大小

# 日志配置文件路径
logging_config: 'config/logging.conf'
```

## 日志配置

日志配置文件位于 `config/logging.conf`，您可以根据需要调整日志级别和格式。默认情况下，日志将同时输出到控制台和 `logs/project.log` 文件。

## 运行步骤

### 1. 训练模型

在命令行中运行以下命令开始训练：

```bash
python src/train.py
```

训练过程中，模型将根据配置文件中的参数进行训练。训练完成后，模型参数将保存至 `outputs/models/model.pdparams`。

### 2. 模型预测

使用训练好的模型对测试集进行预测：

```bash
python src/predict.py
```

预测结果将保存至 `outputs/predictions/submission.csv`，文件格式符合比赛要求，可直接用于提交。

### 3. 结果提交

根据比赛要求，将 `submission.csv` 文件压缩为 `submission.zip`，然后提交至比赛平台。

## 代码说明

- `src/data/`：数据处理模块，包括数据加载、预处理和后处理。
- `src/models/`：模型构建模块，包含模型的定义、保存与加载方法。
- `src/utils/`：工具函数模块，例如设置随机种子等。
- `src/train.py`：模型训练脚本，负责训练流程的控制。
- `src/predict.py`：模型预测脚本，负责生成测试集的预测结果。
- `tests/`：测试模块，包含对数据处理、模型和工具函数的单元测试。

## 数据探索

在 `notebooks/EDA.ipynb` 中，可以进行数据的探索性分析。您可以根据需要添加更多的分析内容，以加深对数据的理解。

## 项目特点

- **模块化设计**：代码按照功能模块划分，清晰明确，方便维护和扩展。
- **配置化管理**：使用 YAML 配置文件集中管理参数，便于实验调参。
- **日志功能完善**：详细的日志记录，方便跟踪程序运行和调试。
- **单元测试**：为关键模块编写了测试用例，保证代码的稳定性和可靠性。

## 注意事项

- 请确保使用的 PaddlePaddle 版本符合比赛要求。
- 运行训练和预测脚本前，请检查数据路径和配置文件是否正确。
- 如需修改模型结构或参数，请在 `config/config.yaml` 中进行调整。

## 参考资料

- [百度飞桨（PaddlePaddle）官网](https://www.paddlepaddle.org.cn/)

