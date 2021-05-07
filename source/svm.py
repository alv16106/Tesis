from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

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

    yy, class_names = pd.factorize(y)
    x_train, x_test, y_train, y_test = train_test_split(data, yy, test_size=0.2, random_state = 203)


    param_grid = {'linearsvc__C': np.logspace(-5, 7, 20)}

    clf = make_pipeline(StandardScaler(), LinearSVC())

    # Búsqueda por validación cruzada
    # ==============================================================================
    grid = GridSearchCV(
            estimator  = clf,
            param_grid = param_grid,
            scoring    = 'accuracy',
            n_jobs     = -1,
            cv         = 3, 
            verbose    = 1,
            return_train_score = True
        )
    _ = grid.fit(X = x_train, y = y_train)

    # Resultados del grid
    # ==============================================================================
    results = pd.DataFrame(grid.cv_results_)
    results.filter(regex = '(param.*|mean_t|std_t)')\
        .drop(columns = 'params')\
        .sort_values('mean_test_score', ascending = False) \
        .head(5)

    print(grid.best_params_, ":", grid.best_score_, grid.scoring)

    model = grid.best_estimator_

    disp = plot_confusion_matrix(model, x_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title('SVC')

    print('SVC')
    print(disp.confusion_matrix)

    plt.show()

    return model

if __name__ == "__main__":
    svm('fixed')