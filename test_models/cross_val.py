import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold, mean_squared_error

if __name__ == '__main__':
    data = pd.read_csv('datasets/felicidad.csv')
    X = data.drop(['country', 'score'], axis = 1)
    y = data['score']
    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, cv = 3, scoring='neg_mean_squared_error')
    print(np.abs(np.mean(score)))
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    mse_values = []
    for train,test in kf.split(data):
        x_train = pd.DataFrame(columns=list(data),index=range(len(train)))
        x_test = pd.DataFrame(columns=list(data),index=range(len(test)))
        y_train = pd.DataFrame(columns=['score'],index=range(len(train)))
        y_test = pd.DataFrame(columns=['score'],index=range(len(test)))
        for i in range(len(train)):
            x_train.iloc[i] = data.iloc[train[i]]
            y_train.iloc[i] = targets.iloc[train[i]]
        for j in range(len(test)):
            x_test.iloc[j] = data.iloc[test[j]]
            y_test.iloc[j] = targets.iloc[test[j]]
        model = DecisionTreeRegressor().fit(x_train,y_train)
        predict = model.predict(x_test)
        mse_values.append(mean_squared_error(y_test,predict))
        
    print("Los tres MSE fueron: ",mse_values)
    print("El MSE promedio fue: ", np.mean(mse_values))