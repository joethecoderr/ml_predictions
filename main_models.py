from utils import Utils
from models import Models

import warnings
warnings.simplefilter("ignore")
def train_heart_disease():
    utils = Utils()
    models = Models()
    ds = utils.load_from_csv('datasets/heart.csv')
    X, y = utils.features_target(ds, ['target'], ['target'])
    models.grid_training(X,y, 'heart')

if __name__ == '__main__':
    # utils = Utils()
    # models = Models()
    # ds = utils.load_from_csv('datasets/felicidad.csv')
    # X, y = utils.features_target(ds, ['score','rank', 'country'],['score'])
    # models.grid_training(X,y, 'felicidad')
    # print(ds)
    train_heart_disease()