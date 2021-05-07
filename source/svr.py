from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import LinearSVR
from helpers import svr_results, plot_c, get_class, print_confusion
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib

import warnings

warnings.filterwarnings('ignore')

def svr(X_train, y_train, X_test, y_test, eps=0.01):
    c_space = np.linspace(0.01, 10)

    test_mae_list = []
    perc_within_eps_list = []
    for c in c_space:
        varied_svr = LinearSVR(epsilon=eps, C=c, fit_intercept=True, max_iter=30000)
        
        varied_svr.fit(X_train, y_train)
        
        test_mae = mean_absolute_error(y_test, varied_svr.predict(X_test))
        test_mae_list.append(test_mae)
        import joblib
        perc_within_eps = 100*np.sum(abs(y_test-varied_svr.predict(X_test)) <= eps) / len(y_test)
        perc_within_eps_list.append(perc_within_eps)

    m = max(perc_within_eps_list)
    inds = [i for i, j in enumerate(perc_within_eps_list) if j == m]
    C = c_space[inds[0]]

    print("best C =", C)
    
    svr_best_C = LinearSVR(epsilon=eps, C=C, fit_intercept=True)
    svr_best_C.fit(X_train, y_train)
    svr_results(y_test, X_test, svr_best_C, eps)

    return svr_best_C


if __name__ == "__main__":
    data = pd.read_pickle('fixed')

    del data['song_id']
    del data['std_valence']
    del data['std_arousal']

    real_y = data['emotion']
    del data['emotion']

    X_train, X_test, y_train, y_test = train_test_split(data, real_y, test_size=0.2, random_state=12)

    Y_arousal_train = X_train['arousal_scaled']
    Y_valence_train = X_train['valence_scaled']

    Y_arousal_test = X_test['arousal_scaled']
    Y_valence_test = X_test['valence_scaled']

    del X_train['valence_scaled']
    del X_train['arousal_scaled']
    del X_test['valence_scaled']
    del X_test['arousal_scaled']
    
#   arousalSVR = svr(X_train, Y_arousal_train, X_test, Y_arousal_test, 0.1)
#    valenceSVR = svr(X_train, Y_valence_train, X_test, Y_valence_test, 0.1)

    arousalSVR = joblib.load('./models/arousal1.pkl')
    valenceSVR = joblib.load('./models/valence1.pkl')

    #predict both
    pred_arousal = arousalSVR.predict(X_test)
    pred_valence = valenceSVR.predict(X_test)

    results = pd.DataFrame()
    results['valence'] = pred_valence
    results['arousal'] = pred_arousal


    predicted_y = results.apply(lambda row: get_class(row.valence, row.arousal), axis = 1)
    
    for a in ['bored', 'happy/excited', 'relaxed', 'angry']:
        print(a, np.count_nonzero(predicted_y == a))

    print(y_test)

    i = 0
    for index, s in enumerate(y_test):
        if s == predicted_y[index]:
            i += 1

    print(i/149, 'accuracy')

    print_confusion(predicted_y, y_test, True, 'matrixes/svr.png')

    # _ = joblib.dump(arousalSVR, './models/arousal1.pkl', compress=9)
    # _ = joblib.dump(valenceSVR, './models/valence1.pkl', compress=9)

