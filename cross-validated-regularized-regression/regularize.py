import sys
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import matplotlib.pyplot as plt


def main(dataset_size, test_proportion):
    diabetes = load_diabetes()
    X = diabetes.data[:dataset_size]
    y = diabetes.target[:dataset_size]

    fig, ax_list = plt.subplots(3, 1, figsize=(8, 6))
    plot_errors_by_lambda(X, y, test_proportion=test_proportion, regression_class=Ridge, ax=ax_list[0])
    plot_errors_by_lambda(X, y, test_proportion=test_proportion, regression_class=Lasso, ax=ax_list[1])
    plot_errors_by_lambda(X, y, test_proportion=test_proportion, regression_class=LinearRegression, ax=ax_list[2])

    plt.tight_layout()
    plt.show()

def fit_regression(X, y, regression_class=LinearRegression, regularization_const=.001):
    '''
        Given a dataset and some solutions (X, y) a regression class (from scikit learn)
        and an Lambda which is required if the regression class is Lasso or Ridge

        X (pandas DataFrame): The data.
        y (pandas DataFrame or Series): The answers.
        regression_class (class): One of sklearn.linear_model.[LinearRegression, Ridge, Lasso]
        Lambda: the regularization_const value (regularization parameter) for Ridge or Lasso. Called alpha by scikit learn
                for interface reasons.

        Return:
            Something.
    '''
    if regression_class is LinearRegression:
        predictor = regression_class()
    else:
        predictor = regression_class(alpha=regularization_const, normalize=True)

    predictor.fit(X, y)

    cross_scores = cross_val_score(predictor, X, y=y, scoring='neg_mean_squared_error')
    cross_scores_corrected = np.sqrt(-1 * cross_scores)  # Scikit learn returns negative vals && we need root

    return (predictor, np.mean(cross_scores_corrected))


def plot_errors_by_lambda(X, y, test_proportion=0.25, lambda_values=None, regression_class=Ridge, ax=None):
    if not lambda_values:
        if regression_class is Lasso:
            lambda_values = np.linspace(0.1, 10)
        else:
            lambda_values = np.logspace(-5, 3)

    # Split the data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion)

    cross_scores = []
    test_rmses = []
    for i, lam in enumerate(lambda_values):
        train_predictor, train_cross_score = fit_regression(X_train, y_train,
            regression_class=regression_class, regularization_const=lam
        )

        cross_scores.append(train_cross_score)

        test_predictions = train_predictor.predict(X_test)
        test_rms_error = np.sqrt(mse(y_test, test_predictions))
        test_rmses.append(test_rms_error)

    # Do the plot
    ax.plot(lambda_values, cross_scores, label='train data cross score')
    ax.plot(lambda_values, test_rmses, label='test data RMSE')
    ax.set_xlabel('lambda')
    ax.set_ylabel('RMSE')
    scale_type = 'log' if regression_class is Ridge else 'linear'
    ax.set_xscale(scale_type)
    ax.legend()
    ax.set_title(str(regression_class.__name__))


if __name__ == '__main__':
    data_slice_size = 150
    test_proportion = .25
    if len(sys.argv) == 3:
        data_slice_size = int(sys.argv[1])
        test_proportion = float(sys.argv[2])

    print "Using {} rows of data and test_proportion: {}".format(data_slice_size, test_proportion)
    main(data_slice_size, test_proportion)
