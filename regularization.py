import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet

if __name__ == '__main__':
    ds = pd.read_csv('datasets/felicidad.csv')
    X = ds[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = ds[['score'] ]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    y_predict_linear = linear.predict(X_test)

    lasso = Lasso(alpha=0.02)
    lasso.fit(X_train, y_train)
    y_predict_lasso = lasso.predict(X_test)

    ridge = Ridge(alpha=1)
    ridge.fit(X_train, y_train)
    y_predict_ridge = ridge.predict(X_test)
    elastic = ElasticNet(alpha = 0.01, random_state=0)
    elastic.fit(X, y)
    y_predict_elastic = elastic.predict(X_test)


    print('Linear Loss: ', mean_squared_error(y_test, y_predict_linear))
    print('Lasso Loss: ', mean_squared_error(y_test, y_predict_lasso))
    print('Ridge Loss: ', mean_squared_error(y_test, y_predict_ridge))
    print('Elastic Loss: ', mean_squared_error(y_test, y_predict_elastic))
    print('='*32)
    print('Coef Linear')
    print(linear.coef_)
    print('Coef Lasso')
    print(lasso.coef_)
    print('Coef Ridge')
    print(ridge.coef_)
    print('Coef Elastic')
    print(elastic.coef_)
    print('Lasso score: ', lasso.score(X_test, y_test))
    print('Linear Score: ', linear.score(X_test, y_test))
    print('Ridge Score: ', ridge.score(X_test, y_test))
    print('Elastic Score: ', elastic.score(X_test, y_test))
