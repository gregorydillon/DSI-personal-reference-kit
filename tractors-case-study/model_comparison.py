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
    # TODO refactor to use the scikit learn version, to be able to more directly compare the
    # linear regression model to the other models found here.
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


def cross_validate_best_known():
    tractor_data = pd.read_csv('data/train.csv')
    tractor_data = cln.clean_all(tractor_data)
    X = tractor_data
    y = tractor_data.pop('SalePrice')

    rf = RandomForestRegressor(max_features=2, min_samples_split=4, n_estimators=50, min_samples_leaf=2)
    gb = GradientBoostingRegressor(loss='quantile', learning_rate=0.0001, n_estimators=50, max_features='log2', min_samples_split=2, max_depth=1)
    ada_tree_backing = DecisionTreeRegressor(max_features='sqrt', splitter='random', min_samples_split=4, max_depth=3)
    ab = AdaBoostRegressor(ada_tree_backing, learning_rate=0.1, loss='square', n_estimators=1000)

    validate.cross_v_scores([rf, gb, ab], X, y)
    # RandomForestRegressor -- RMLSE: -0.596797712098, R2: 0.0272065373946
    # GradientBoostingRegressor -- RMLSE: -0.996134592541, R2: -2.37202164829
    # AdaBoostRegressor -- RMLSE: -0.706385708459, R2: -0.103966980393


def grid_search_regressors():
    tractor_data = pd.read_csv('data/train.csv')
    tractor_data = cln.clean_all(tractor_data)
    X = tractor_data
    y = tractor_data.pop('SalePrice')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)  # Leave out 10% for testing later

    rf_grid, rf_model = random_forest_grid_search()
    gb_grid, gb_model = gradient_boost_grid_search()
    ab_grid, ab_model = ada_boost_tree_grid_search()

    rf_gs = validate.grid_search(rf_model, rf_grid, X_train, y_train)
    print 'best parameters:', rf_gs.best_params_
    # best parameters: {'max_features': 'log2', 'min_samples_split': 4, 'n_estimators': 50, 'min_samples_leaf': 2}

    gb_gs = validate.grid_search(gb_model, gb_grid, X_train, y_train)
    print 'best parameters:', gb_gs.best_params_
    # best parameters {'loss': 'quantile', 'learning_rate': 0.0001, 'n_estimators': 50, 'max_features': 'log2', 'min_samples_split': 2, 'max_depth': 1}

    ab_gs = validate.grid_search(ab_model, ab_grid, X_train, y_train)
    print 'best parameters:', ab_gs.best_params_
    # best parameters: {'loss': 'square', 'base_estimator__min_samples_split': 4, 'base_estimator__max_depth': 3, 'learning_rate': 0.1, 'base_estimator__max_features': 'sqrt', 'n_estimators': 1000, 'base_estimator__splitter': 'random'}

    best_forest = rf_gs.best_estimator_
    best_gradient_boost = gb_gs.best_estimator_
    best_ada_boost = ab_gs.best_estimator_
    estimators = [best_forest, best_gradient_boost, best_ada_boost]
    best_score, best_regressor = select_best_regressor(estimators,  X_train, X_test, y_train, y_test)

    print "Best Regressor: {}".format(best_regressor.__class__.__name__)
    print "Score: {best_score}"


def select_best_regressor(estimators, X_train, X_test, y_train, y_test):
    scores = []
    for e in estimators:
        e.fit(X_train, y_train)
        s = validate.root_mean_log_squared_error(e, X_test, y_test)
        scores.append((s, e))

    return min(scores)


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
        'learning_rate': [.0001, .001, .01, .1, 1],
        'n_estimators': [50, 100, 1000, 10000],
        'max_depth': [1, 3],
        'min_samples_split': [2, 4, 10],
        'max_features': ['sqrt', 'log2'],
    }
    gb = GradientBoostingRegressor()

    return gradient_boost_grid, gb


def ada_boost_tree_grid_search():
    ada_boost_tree_grid = {
        'base_estimator__max_features': ['sqrt'],
        'base_estimator__splitter': ['best', 'random'],
        'base_estimator__min_samples_split': [2, 4],
        'base_estimator__max_depth': [1, 3],
        'n_estimators': [50, 100, 1000],
        'learning_rate': [.001, .01, .1],
        'loss': ['linear', 'square', 'exponential']
    }
    abr = AdaBoostRegressor(DecisionTreeRegressor())

    return ada_boost_tree_grid, abr


if __name__ == '__main__':
    # grid_search_regressors()
    cross_validate_best_known()
