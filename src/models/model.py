import paddle
import paddle.nn as nn
import logging
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import joblib  # 用于保存和加载模型
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm  # 导入 tqdm 库
from xgboost.callback import TrainingCallback  # XGBoost 回调基类
from lightgbm.callback import CallbackEnv  # LightGBM 回调环境

logger = logging.getLogger(__name__)

class LinearRegressionModel(nn.Layer):
    """线性回归模型"""
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

class MLPModel(nn.Layer):
    """多层感知机模型"""
    def __init__(self, input_size, hidden_sizes):
        super(MLPModel, self).__init__()
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        layers.append(nn.Linear(last_size, 1))  # 输出层，节点数为1
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class FullyConnectedModel(nn.Layer):
    """全连接神经网络模型"""
    def __init__(self, input_size, hidden_sizes):
        super(FullyConnectedModel, self).__init__()
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # 添加Dropout层
            last_size = size
        layers.append(nn.Linear(last_size, 1))  # 输出层，节点数为1
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class XGBoostProgressBar(TrainingCallback):
    """XGBoost 进度条回调"""
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc='XGBoost Training')

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False  # 返回 False 以继续训练

    def after_training(self, model):
        self.pbar.close()

class XGBoostModel:
    """XGBoost 模型封装"""
    def __init__(self, params):
        self.params = params
        self.n_estimators = params.get('n_estimators', 100)
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 1.0),
            colsample_bytree=params.get('colsample_bytree', 1.0),
            eval_metric=params.get('eval_metric', 'logloss'),
            use_label_encoder=False
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        callbacks = [XGBoostProgressBar(self.n_estimators)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
            callbacks=callbacks
        )

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)
        logger.info(f"XGBoost 模型已保存至 {path}")

    def load(self, path):
        self.model = joblib.load(path)
        logger.info(f"XGBoost 模型已从 {path} 加载")

    def tune_parameters(self, X_train, y_train):
        """参数调优方法"""
        # 定义参数网格
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.1, 0.05, 0.01],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 5, 10]
        }

        xgb_model = xgb.XGBClassifier(eval_metric='logloss')

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=3,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        logger.info(f"最佳参数: {grid_search.best_params_}")
        return grid_search.best_params_

class LightGBMProgressBar:
    """LightGBM 进度条回调"""
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc='LightGBM Training')

    def __call__(self, env: CallbackEnv):
        if env.iteration == 0:
            self.pbar.reset(total=env.params['n_estimators'])
        self.pbar.update(1)
        if env.iteration + 1 == env.params['n_estimators']:
            self.pbar.close()
        return

class LightGBMModel:
    """LightGBM 模型封装"""
    def __init__(self, params):
        self.params = params
        self.n_estimators = params.get('n_estimators', 100)
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=params.get('max_depth', -1),
            learning_rate=params.get('learning_rate', 0.1),
            num_leaves=params.get('num_leaves', 31),
            subsample=params.get('subsample', 1.0),
            colsample_bytree=params.get('colsample_bytree', 1.0)
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            eval_names.append('valid')
        callbacks = [LightGBMProgressBar(self.n_estimators)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            verbose=False,
            callbacks=callbacks
        )

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)
        logger.info(f"LightGBM 模型已保存至 {path}")

    def load(self, path):
        self.model = joblib.load(path)
        logger.info(f"LightGBM 模型已从 {path} 加载")

def save_paddle_model(model, path):
    """保存 Paddle 模型"""
    paddle.save(model.state_dict(), path)
    logger.info(f"Paddle 模型已保存至 {path}")

def load_paddle_model(model, path):
    """加载 Paddle 模型"""
    state_dict = paddle.load(path)
    model.set_state_dict(state_dict)
    logger.info(f"Paddle 模型已从 {path} 加载")
