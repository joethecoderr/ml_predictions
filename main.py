from utils import Utils
from models import Models

if __name__ == '__main__':
    utils = Utils()
    models = Models()
    ds = utils.load_from_csv('datasets/felicidad.csv')
    X, y = utils.features_target(ds, ['score','rank', 'country'],['score'])
    models.grid_training(X,y)
    print(ds)