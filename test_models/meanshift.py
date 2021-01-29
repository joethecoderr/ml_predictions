import pandas as pd
from sklearn.cluster import MeanShift

if __name__ == '__main__':
    ds = pd.read_csv('datasets/candy.csv')
    X = ds.drop('competitorname', axis = 1)
    meanshift = MeanShift()
    meanshift.fit(X)
    print('='*64)
    print(meanshift.cluster_centers_)
    ds['meanshift'] = meanshift.labels_
    print('='*64)
    print(ds)