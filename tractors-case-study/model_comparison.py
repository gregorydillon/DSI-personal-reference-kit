'''
This file is a collection of models which will all be trained on the cleaned
tractor data. It's also a collection of tools for comparing such models, and
fine tuning their paramters using tactics like GridSearchCV.

Once optimal models have been found within each model type the optimal versions
of these models are cross validated, and the results compared across model types.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import statsmodels.formula.api as smf


import validate
import cleaner as cln


def regression_formula_model(tractor_data):
    '''
    Because this model is a StatsModels based tool, more work has to be hand rolled.
    There are specific validations for using this model, which LEAVES the column being
    predicted IN the dataset -- since it's specified in the formula
    '''
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
    model = smf.ols(formula=favorite_so_far, data=tractor_data)

    return model, col_to_predict


def grid_search_regressors():
    tractor_data = pd.read_csv('data/train.csv')
    tractor_data = cln.clean_all(tractor_data)

    # Just select the data we decided previously we care about
    # Room for improvement here
    X = tractor_data.filter([
        'Modernity',
        'MachineHoursCurrentMeter',
        'ProductGroup',
        'Enclosure',
        'Hydraulics',
        '_units',
        '_measurement'
    ])
    y = tractor_data.pop('SalePrice')

    # DUMMIFY, TODO: move to cleanup?
    to_dummy = [
        'Modernity',
        'ProductGroup',
        'Enclosure',
        'Hydraulics',
        '_units'
    ]
    X = pd.get_dummies(X, columns=to_dummy)

    # Imput NaN's TODO: Move to cleanup?
    X.MachineHoursCurrentMeter[
        np.isnan(X.MachineHoursCurrentMeter)
    ] = np.mean(X.MachineHoursCurrentMeter[~np.isnan(X.MachineHoursCurrentMeter)])

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)  # Leave out 10% for testing later

    # SMALL INTENTIONALLY TO TEST SHIIIIIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.7)  # Leave out 10% for testing later

    rf_grid, rf_model = random_forest_grid_search()
    gb_grid, gb_model = gradient_boost_grid_search()
    ab_grid, ab_model = ada_boost_tree_grid_search()

    rf_gs = validate.grid_search(rf_model, rf_grid, X_train, y_train)
    print 'best parameters:', rf_gs.best_params_

    gb_gs = validate.grid_search(gb_model, gb_grid, X_train, y_train)
    print 'best parameters:', gb_gs.best_params_

    ab_gs = validate.grid_search(ab_model, ab_grid, X_train, y_train)
    print 'best parameters:', ab_gs.best_params_


def random_forest_grid_search():
    random_forest_grid = {
        'n_estimators': [50, 100, 1000],
        'max_features': ['sqrt', 'log2', 'auto'],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
    }
    rf = RandomForestRegressor()

    return random_forest_grid, rf


def gradient_boost_grid_search():
    gradient_boost_grid = {
        'loss': ['ls', 'lad', 'huber', 'quantile'],
        'learning_rate': [.001, .01, .1],
        'n_estimators': [100, 500, 1000],
        'max_depth': [1, 3],
        'min_samples_split': [2, 4],
        'max_features': ['sqrt'],
    }
    gb = GradientBoostingRegressor()

    return gradient_boost_grid, gb


def ada_boost_tree_grid_search():
    ada_boost_tree_grid = {
        'base_estimator__max_features': ['sqrt', 'log2', 'auto'],
        'base_estimator__criterion': ['gini', 'entropy'],
        'base_estimator__splitter': ['best', 'random'],
        'base_estimator__min_samples_split': [2, 4],
        'base_estimator__max_depth': [1, 3, 5],
        'n_estimators': [100, 500, 1000],
        'learning_rate': [.001, .01, .1],
        'loss': ['linear', 'square', 'exponential']
    }
    abr = AdaBoostRegressor(DecisionTreeRegressor())

    return ada_boost_tree_grid, abr

if __name__ == '__main__':
    grid_search_regressors()
