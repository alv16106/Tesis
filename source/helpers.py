from sklearn.metrics import mean_absolute_error 
import matplotlib.pyplot as plt
import numpy as np

def svr_results(y_test, X_test, fitted_svr_model, eps):
    
    print("C: {}".format(fitted_svr_model.C))
    print("Epsilon: {}".format(fitted_svr_model.epsilon))
    
    print("Intercept: {:,.3f}".format(fitted_svr_model.intercept_[0]))
    print("Coefficient: {:,.3f}".format(fitted_svr_model.coef_[0]))
    
    mae = mean_absolute_error(y_test, fitted_svr_model.predict(X_test))
    print("MAE = ${:,.2f}".format(1000*mae))
    
    perc_within_eps = 100*np.sum(y_test - fitted_svr_model.predict(X_test) < eps) / len(y_test)
    print("Percentage within Epsilon = {:,.2f}%".format(perc_within_eps))

    print(y_test - fitted_svr_model.predict(X_test))

def plot_c(c_space, perc_within_eps_list, test_mae_list):
    fig, ax1 = plt.subplots(figsize=(12,7))

    color='green'
    ax1.set_xlabel('C')
    ax1.set_ylabel('% within Epsilon', color=color)
    ax1.scatter(c_space, perc_within_eps_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color='blue'
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Test MAE', color=color)  # we already handled the x-label with ax1
    ax2.scatter(c_space, test_mae_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)


    plt.show()
