import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    ds = pd.read_csv('datasets/felicidad.csv')
    X = ds.drop(['country', 'rank', 'score'], axis = 1)
    y = ds['score']
    reg = RandomForestRegressor()
    params = {
        'n_estimators' : range(4,16),
        'criterion' : ['mse', 'mae'],
        'max_depth' : range(2,11)
    }
    rand_est = RandomizedSearchCV(reg, params, n_iter=10, cv = 3, scoring = 'neg_mean_absolute_error')
    rand_est.fit(X,y)
    print('Best estimator: ', rand_est.best_estimator_)
    print('Best params: ', rand_est.best_params_)
    print('Prediction: ', rand_est.predict(X.loc[[0]]))