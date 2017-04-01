import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

    model, scores = validate_formula(favorite_so_far, tractor_data, col_to_predict)
    print "LRMSE: {}".format(np.mean(scores))


def validate_formula(formula, training_data, column_being_predicted, cross_val_n=3, validation_size=.10):
    '''
        Accept a formula in the StatsModels.formula.api style, some training data and
        some test values that must match the value being predicted by the formula.

        returns: trained_model, cross_scores
    '''
    cross_val_scores = []
    for _ in xrange(cross_val_n):
        X_train, X_test, _, _ = train_test_split(
            training_data,
            training_data[column_being_predicted],
            test_size=validation_size
        )

        model = smf.ols(formula=formula, data=X_train).fit()
        test_values = X_test[column_being_predicted]
        score = root_mean_log_squared_error(model, X_test, test_values)
        cross_val_scores.append(score)

    return (model, cross_val_scores)


def grid_search(model, feature_dict, train_x, train_y):
    gscv = GridSearchCV(
        model,
        feature_dict,
        n_jobs=-1,
        verbose=True,
        scoring=root_mean_log_squared_error
    )

    gscv.fit(train_x, train_y)
    return gscv


def cross_v_scores(regressors, training_data, training_targets):
    for r in regressors:
        mse, r2 = cross_validate(r, training_data, training_targets)
        print('{} -- MSE: {}, R2: {}'.format(r.__class__.__name__, mse, r2))


def cross_validate(estimator, training_data, test_targets):
    mse = cross_val_score(estimator, X=training_data, y=test_targets, scoring=root_mean_log_squared_error)
    r2 = cross_val_score(estimator, X=training_data, y=test_targets, scoring='r2')

    return (-1 * np.mean(mse), np.mean(r2))


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
