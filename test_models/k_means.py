import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot  as plt
import seaborn as sns

if __name__ == '__main__':
    ds = pd.read_csv('datasets/candy.csv')
    X = ds.drop('competitorname', axis = 1)
    kmeans = MiniBatchKMeans(n_clusters = 4, batch_size=8)
    kmeans.fit(X)
    print("Total de centros: ", len(kmeans.cluster_centers_))
    print('='*64)
    ds['group'] = kmeans.predict(X)
    print(ds)
    sns.pairplot(ds[['sugarpercent','pricepercent','winpercent','group']], hue = 'group')
    plt.savefig('plots/pairplot.png')