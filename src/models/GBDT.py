import paddle
from paddle import nn
from sklearn.ensemble import GradientBoostingClassifier

class GBDTModel:
    def __init__(self, params):
        self.model = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth']
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
