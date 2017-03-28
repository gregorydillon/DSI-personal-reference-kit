from validate import test_formula
import pandas as pd
import cleaner as cln

tractor_data = pd.read_csv('data/train.csv')
tractor_data = cln.clean_all(tractor_data)

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

simple = "SalePrice ~ Modernity"

# print tractor_data.info()
# print tractor_data.description()
# print tractor_data.head(5)

# model, score = test_formula(simple, tractor_data, col_to_predict)
# print score
