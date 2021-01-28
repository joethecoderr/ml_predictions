import pandas as pd
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

if __name__ == '__main__':
    ds = pd.read_csv('datasets/felicidad_corrupt.csv')
    X = ds.drop(['country', 'score'], axis = 1)
    y = ds['score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)
    estimators = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        print('-'*70)
        print(r'{name} MSE: ', mean_squared_error(y_test, predictions))
        print(f'{name} Score is: ', estimator.score(X_test, y_test))
        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title('Predicted VS Real')
        plt.scatter(y_test, predictions)
        plt.plot(predictions, predictions,'r--')
        plt.savefig(f'robusts/plot_{name}.png')
