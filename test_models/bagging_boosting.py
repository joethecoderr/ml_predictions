import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore")

def plot_gradient_boosting():
    n_estimators=range(10,200,10)
    total_accuracy=[]
    for i in n_estimators:
        boost = GradientBoostingClassifier(n_estimators=i).fit(X_train, y_train)
        boost_pred = boost.predict(X_test)

        total_accuracy.append(accuracy_score(y_test, boost_pred))

    plt.plot(n_estimators, total_accuracy)
    plt.xlabel('Estimators')
    plt.ylabel('Accuracy')
    plt.savefig('robusts/Boost.png')

    print(np.array(total_accuracy).max())

if __name__ == '__main__':
    dt = pd.read_csv('datasets/heart.csv')
    X = dt.drop(['target'], axis=1)
    y = dt['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35, random_state=42)
    estimators = {
        'knn_class': KNeighborsClassifier(n_neighbors=4),
        'bag_class' : BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=4), n_estimators=50),
        'LinearSVC': LinearSVC(),
        'SVC': SVC(),
        'SGDClassifiser': SGDClassifier(),
        'Descission Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=125)
    }
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        prediction = estimator.predict(X_test)
        print('='*64)
        print(f'{name}')
        print('Accuracy: ' , accuracy_score(y_test, prediction))


    """cp = chest pain type range(0,3)
       trestbps = resting blood pressure
       chol = cholestorel in mg/dl
       fbs = fasting blood sugar if fbs > 120 then true
       restecg = resting electrocardiograhic results (1,0)
       exercise induced angina
       thalach = max heart rate achieved
       oldpeak = ST depression induced by exercise (continuous value)
       slope range(0,2)
       ca = number of major vessels range(0,3)
       thal = 1 == normal, 2 == fixed defect, 3 == reversable defect  (blood desorder, not enough hemogoblin)
    """