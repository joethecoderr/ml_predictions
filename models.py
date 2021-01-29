import pandas as pd
import numpy as np
from sklearn.svm import SVR, LinearSVC
from sklearn.ensemble import GradientBoostingRegressor, BaggingClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from utils import Utils

import warnings
warnings.simplefilter("ignore")
class Models:
    def __init__(self):
        self.reg = {
            # 'SVR': SVR(),
            # 'GRADIENT': GradientBoostingRegressor(),
             'FORREST' : RandomForestRegressor(),
             'LinearSVC': LinearSVC(),
             'GradientClass' : GradientBoostingClassifier()
        }
        self.params = {
            # 'SVR' : {
            #     'kernel' : ['linear', 'poly', 'rbf'],
            #     'gamma' : ['auto', 'scale'],
            #     'C' : [1,5,10]
            # },
            # 'GRADIENT' : {
            #     'loss' : ['ls', 'lad'],
            #     'learning_rate' : [0.01, 0.05, 0.1]
            # },
            'FORREST' : {
                'n_estimators' : range(6,11),
                'criterion' : ['mse', 'mae'],
                'max_depth' : range(4,11)
            },
            'LinearSVC': {
                'max_iter' : [1000],
            },
            'GradientClass' : {
                'n_estimators' : [125],
                'learning_rate': [0.01, 0.05, 0.1],
                'criterion': ['friedman_mse', 'mse']
            }

        }
    def grid_training(self, X, y, dataset_name):
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            grid_reg = GridSearchCV(reg, self.params[name], cv = 3)
            grid_reg.fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)
            if score > best_score:
                best_score = score
                best_model = grid_reg.best_estimator_
        utils = Utils()
        utils.model_export(best_model, best_score, dataset_name + name)


