from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from keras.utils import np_utils

import numpy as np
import pandas as pd


def svm(dataset):
    data = pd.read_pickle(dataset)

    del data['song_id']
    del data['valence_scaled']
    del data['arousal_scaled']
    del data['std_valence']
    del data['std_arousal']

    y = np.array(data.emotion.tolist())

    del data['emotion']

    yy, _ = pd.factorize(y)
    x_train, x_test, y_train, y_test = train_test_split(data, yy, test_size=0.2, random_state = 203)

    param_grid = {'svc__C': np.logspace(-5, 7, 20)}

    clf = make_pipeline(StandardScaler(), SVC(kernel= "rbf", gamma='scale'))

    # Búsqueda por validación cruzada
    # ==============================================================================
    grid = GridSearchCV(
            estimator  = clf,
            param_grid = param_grid,
            scoring    = 'accuracy',
            n_jobs     = -1,
            cv         = 3, 
            verbose    = 0,
            return_train_score = True
        )
    _ = grid.fit(X = x_train, y = y_train)

    # Resultados del grid
    # ==============================================================================
    resultados = pd.DataFrame(grid.cv_results_)
    resultados.filter(regex = '(param.*|mean_t|std_t)')\
        .drop(columns = 'params')\
        .sort_values('mean_test_score', ascending = False) \
        .head(5)



    # Mejores hiperparámetros por validación cruzada
    # ==============================================================================
    print("----------------------------------------")
    print("Mejores hiperparámetros encontrados (cv)")
    print("----------------------------------------")
    print(grid.best_params_, ":", grid.best_score_, grid.scoring)

    modelo = grid.best_estimator_

if __name__ == "__main__":
    svm('fixed')