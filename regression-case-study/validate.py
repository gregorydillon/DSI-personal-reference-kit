import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import cleaner as cln


def main():
    # Best so far -- .499 error
    # Credit to Vy Nguyen
    col_to_predict = 'SalePrice'
    favorite_so_far = '''
    SalePrice ~ YearMade +
            Modernity +
            MachineHoursCurrentMeter +
            ProductGroup +
            Enclosure +
            Hydraulics +
            _units*_measurement
    '''
    tractor_data = pd.read_csv('data/train.csv')
    tractor_data = cln.clean_all(tractor_data)

    model, score = test_formula(favorite_so_far, tractor_data, col_to_predict)
    print "LRMSE: {}".format(score)


def test_formula(formula, training_data, column_being_predicted):
    '''
        Accept a formula in the StatsModels.formula.api style, some training data and
        some test values that must match the value being predicted by the formula.

        returns: trained_model, cross_scores, test_score
    '''
    # Formula requires that we include the field being predicted in the X data both times
    # So we just include the name of the column which is kind of nice honestly.
    X_train, X_test, _, _ = train_test_split(training_data, training_data[column_being_predicted], test_size=.10)
    model = smf.ols(formula=formula, data=X_train).fit()

    test_values = X_test[column_being_predicted]
    score = root_mean_log_squared_error(model, X_test, test_values)

    return (model, score)


def root_mean_log_squared_error(model, X, y):
    '''
        compute the log-root-mean-squared error metric. This function
        doesn't punish being WAY off on some values as highly.
    '''
    predictions = model.predict(X)
    log_diff = np.log(predictions + 1) - np.log(y + 1)
    return np.sqrt(np.mean(log_diff ** 2))


if __name__ == '__main__':
    main()
