import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from zipfile import ZipFile

from performotron import Comparer
from cleanup import create_and_norm_categorical


class RMLSEComparer(Comparer):
    def score(self, predictions):
        log_diff = np.log(predictions+1) - np.log(self.target+1)
        return np.sqrt(np.mean(log_diff**2))


def score_that_shit(predictions):
    predictions = properly_index_y(predictions)
    predictions.set_index('SalesID')
    test_solution = pd.read_csv('data/do_not_open/test_soln.csv')
    test_solution.set_index('SalesID')
    c = RMLSEComparer(test_solution.SalePrice)

    return c.score(predictions.SalePrice)


def properly_index_y(y):
    y = y.reset_index()
    y['SalePrice'] = y[0]
    y['SalesID'] = y['index']
    del y['index']
    del y[0]

    return y


def score_from_formula(formula, training_data, test_data):
    est = smf.ols(formula=formula, data=training_data).fit()
    y = est.predict(test_data)
    return score_that_shit(y)


zf = ZipFile('data/Train.zip')
df = pd.read_csv('data/Train.csv')
cleaned = create_and_norm_categorical(df)

zf_tst = ZipFile('data/Test.zip')
df_tst = pd.read_csv('data/test.csv')
df_tst_clean = create_and_norm_categorical(df_tst)

# Best so far -- .499 error
f1 = '''
SalePrice ~ YearMade +
        Modernity +
        MachineHoursCurrentMeter +
        ProductGroup +
        fiProductClassDesc +
        Enclosure +
        Hydraulics +
        _units*_measurement
'''

s1 = score_from_formula(f1, cleaned, df_tst_clean)


f = '''
SalePrice ~
        Modernity +
        MachineHoursCurrentMeter +
        ProductGroup +
        fiProductClassDesc +
        Enclosure_Reduced +
        Hydraulics +
        _units*_measurement
'''
s = score_from_formula(f, cleaned, df_tst_clean)
