'''
This file is a collection of models which will all be trained on the cleaned
tractor data. It's also a collection of tools for comparing such models, and
fine tuning their paramters using tactics like GridSearchCV.

Once optimal models have been found within each model type the optimal versions
of these models are cross validated, and the results compared across model types.
'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor


def regression_formula_model():
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
    model = smf.ols(formula=formula, data=tractor_data)

    return model, model_to_predict


def random_forest_grid_search():
    random_forest_grid = {
        'n_estimators': [10, 20, 40, 100, 1000],
        'max_features': ['sqrt', 'log2', 'auto'],
        'min_samples_split': [2, 4, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
    }
    rf = RandomForestRegressor()

    return random_forest_grid, rf


def gradient_boost_grid_search():
    gradient_boost_grid = {
        'loss': ['ls', 'lad', 'huber', 'quantile'],
        'learning_rate': [.001, .01, .1, 1, 10],
        'n_estimators': [10, 20, 40, 100, 1000],
        'max_depth': [1, 3, 5],
        'min_samples_split': [2, 4, 10],
        'subsample': [1, .9, .5],
        'max_features': ['sqrt', 'log2', 'auto'],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    gb = GradientBoostingRegressor()

    return gradient_boost_grid, gb


def ada_boost_tree_grid_search():
    ada_boost_tree_grid = {
        'base_estimator__max_features': ['sqrt', 'log2', 'auto'],
        'base_estimator__criterion': ['gini', 'entropy'],
        'base_estimator__splitter': ['best', 'random'],
        'base_estimator__min_samples_split': [2, 4, 10],
        'base_estimator__min_samples_leaf': [1, 2, 4],
        'base_estimator__max_depth': [1, 3, 5],
        'n_estimators': [10, 20, 40, 100, 1000],
        'learning_rate': [.001, .01, .1, 1, 10],
        'loss': ['linear', 'square', 'exponential']
    }
    abr = AdaBoostRegressor(DecisionTreeRegressor())

    return ada_boost_tree_grid, abr
