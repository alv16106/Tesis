from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.svm import LinearSVR
from helpers import svr_results, plot_c
import pandas as pd
import numpy as np

def svr(X_train, y_train, X_test, y_test, eps=0.01):
    c_space = np.linspace(0.01, 10)

    test_mae_list = []
    perc_within_eps_list = []
    for c in c_space:
        varied_svr = LinearSVR(epsilon=eps, C=c, fit_intercept=True, max_iter=10000)
        
        varied_svr.fit(X_train, y_train)
        
        test_mae = mean_absolute_error(y_test, varied_svr.predict(X_test))
        test_mae_list.append(test_mae)
        
        perc_within_eps = 100*np.sum(abs(y_test-varied_svr.predict(X_test)) <= eps) / len(y_test)
        perc_within_eps_list.append(perc_within_eps)

    m = max(perc_within_eps_list)
    inds = [i for i, j in enumerate(perc_within_eps_list) if j == m]
    C = c_space[inds[0]]

    print("best C =", C)
    
    svr_best_C = LinearSVR(epsilon=eps, C=C, fit_intercept=True)
    svr_best_C.fit(X_train, y_train)
    svr_results(y_test, X_test, svr_best_C, eps)


if __name__ == "__main__":
    data = pd.read_pickle('final')

    Y_arousal = data['arousal_scaled']
    Y_valence = data['valence_scaled']

    del data['song_id']
    del data['valence_scaled']
    del data['arousal_scaled']
    del data['std_valence']
    del data['std_arousal']

    X_train, X_test, y_train, y_test = train_test_split(data, Y_arousal, test_size=0.2, random_state=12)

    svr(X_train, y_train, X_test, y_test, 0.1)

