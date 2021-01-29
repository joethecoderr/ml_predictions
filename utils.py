import pandas as pd
import joblib

class Utils:
    def __init__(self):
        pass

    def load_from_csv(self, path):
        return pd.read_csv(path)

    def load_from_mysql(self):
        pass

    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X,y

    def model_export(self, clf, score, dataset_name):
        print(score)
        joblib.dump(clf, f'./models/best_model_score_{round(score,4)}_{dataset_name}.pkl')