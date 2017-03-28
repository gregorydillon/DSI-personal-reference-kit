import pandas as pd
import re
from zipfile import ZipFile

# Configuration Parameters -- This is not the best design pattern
# TODO: Refactor to command-line options
UNKNOWN_VALUE = 'Unknown'
INPUT_NONE_VALUES = ['None or Unspecified', 'Unspecified', '1000', 'nan']
KNOWN_INVALID_ENTRY_VALUES = ['#NAME?']
NUMERICAL_CATEGORICAL_COLUMNS = [
    'ModelID', 'datasource', 'YearMade', 'auctioneerID',
]

def main():
    data = None #  Fix
    data = clean_all(data, NUMERICAL_CATEGORICAL_COLUMNS)


def clean_all(df, fields_to_clone):
    '''
        Given a DataFrame and a list of columns to clone, return a copy
        of the dataframe with copies of that field as strings, for categorical
        classification. Additionally, perform several operations that fix known errors
        specific to the Tractors dataset. (Though there are many more yet)

        df (DataFrame): The dataframe
        fields_to_clone (list[str]): the name of the columns to be cloned.

        return (DataFrame): a df with the new, better, expanded data
    '''
    copy = df.copy()
    for col_name in NUMERICAL_CATEGORICAL_COLUMNS:
        copy[col_name + '_categorical'] = copy[col_name].astype(str)

    copy = merge_all_none_types(copy)

    return copy


def merge_all_none_types(df):
    '''
    Using Configuration parameters above, take all the
    '''
    copy = df.copy()
    for col_name in df:
        series = copy[col_name]
        if series.dtype in ['string', 'object']:
            new_series = series.apply(none_mapper)
            copy[col_name] = new_series

    return copy


def none_mapper(value):
    if pd.isnull(value):
        return UNKNOWN_VALUE
    elif value == '':
        return UNKNOWN_VALUE
    elif value in INPUT_NONE_VALUES:
        return UNKNOWN_VALUE
    elif value in KNOWN_INVALID_ENTRY_VALUES:
        return UNKNOWN_VALUE

    return value


def combine_10_inch_tire_size(value):
    if value == '10 inch':
        return '10"'

    return value


def enclosure_ac_map(value):
    '''
        Maps the Enclosure column to an educated guess about the Air Conditioning status
        of the tractor's enclosure.
    '''
    if 'AC' in ['EROPS w AC', 'EROPS AC']:
        return 'Has AC'
    elif value in ['OROPS', 'OROPS', 'EROPS', 'NO ROPS']:
        return 'Likely No AC'
    else:
        return 'AC STATUS UNKNOWN'


def year_buckets_cat_map(val):
    '''
        Map a single year value to the 20 year buckets between 1940 and 2000,
        mark a known error (value 1000) as the configured UNKNOWN_VALUE.
    '''
    if val == 1000:
        return UNKNOWN_VALUE
    elif val < 1940:
        return "Before 1940"
    elif val < 1960:
        return "1940 - 1960"
    elif val < 1980:
        return "1960 - 1980"
    elif val < 2000:
        return "1980 - 2000"
    else:
        return "After 2000"


def product_descr_to_units(desc_value):
    '''
        Extract the values from fiProductClassDesc using regex and fetch
        the units from the range provided.

        All Credit to Alan Jennings
    '''
    result_name = re.search('[^0-9+]+$', desc_value)  #grab units from the tail..
    if result_name:
        units = result_name.group(0)
        return units
    else:
        return UNKNOWN_VALUE


def product_descr_to_mean(desc_value):
    '''
        Extract the values from fiProductClassDesc using regex and create
        numerical mean's from the range provided.

        All Credit to Alan Jennings
    '''
    result = re.search('([0-9\.]+) to ([0-9\.]+)', desc_value)
    if result:
        value = (float(result.group(1)) + float(result.group(2))) / 2
        return value
    else:
        #for cases "16.0 + Ft Standard Digging Depth"
        result2 = re.search('([0-9\.]+)\s?\+', desc_value)
        if result2:
            value = result2.group(1)
            return float(value)
        else:
            return 0


def apply_all_specific_transforms(df):
    '''
        Performs several cleaning operations that merge specifically known problems in the
        data. See the applied functions. Operation is in place, returns the provided dataframe.
    '''
    clean_data['Tire_Size'] = clean_data['Tire_Size'].apply(combine_10_inch_tire_size)
    clean_data['Modernity'] = clean_data['YearMade'].apply(year_buckets_cat_map)
    clean_data['Enclosure_Reduced'] = clean_data['Enclosure'].apply(enclosure_ac_map)
    clean_data['_units'] = clean_data['fiProductClassDesc'].apply(product_descr_to_units)
    clean_data['_measurement'] = clean_data['fiProductClassDesc'].apply(product_descr_to_mean)

    return clean_data