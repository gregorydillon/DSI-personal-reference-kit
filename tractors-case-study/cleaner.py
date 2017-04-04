import numpy as np
import pandas as pd
import re

# Configuration Parameters -- This is not the best design pattern
# TODO: Refactor to command-line options
OUTPUT_UNKNOWN_VALUE = 'Unknown'
INPUT_NONE_VALUES = ['None or Unspecified', 'Unspecified', 'nan', '', 'Unknown']
KNOWN_INVALID_ENTRY_VALUES = ['#NAME?']
NUMERICAL_CATEGORICAL_COLUMNS = [
    'ModelID', 'datasource', 'YearMade', 'auctioneerID',
]


def clean_all(df, apply_all_specific_transforms=None):
    '''
        Given a DataFrame use NUMERICAL_CATEGORICAL_COLUMNS to return a copy
        of the dataframe with copies of that field as strings for categorical
        classification. Additionally, perform several operations that fix known errors
        specific to the Tractors dataset. (Though there are many more yet)

        df (DataFrame): The dataframe

        return (DataFrame): a df with the new, better, expanded data
    '''
    copy = df.copy()
    for col_name in NUMERICAL_CATEGORICAL_COLUMNS:
        copy[col_name + '_categorical'] = copy[col_name].astype(str)

    copy = merge_all_none_types(copy)

    # TRACTOR specific
    # TODO: Don't allow default function.
    apply_all_specific_transforms = apply_all_specific_transforms or apply_all_specific_tractor_transforms
    apply_all_specific_transforms(copy)

    # THE ORDER OF THESE OPERATIONS CAN BE IMPORTANT
    copy = remove_uninformative_rows(copy)
    copy = remove_uninformative_columns(copy)
    copy = impute_median_all(copy)
    copy = dummify(copy)

    return copy


def remove_uninformative_rows(df):
    '''
        Given a dataframe assumed to have been loaded from our Tractor Data -- remove rows that don't provide
        much information. Specifically these are rows that have been determined to have "unknown" values in
        columns we have decided are important to our regression models.

        returns DataFrame -- the input dataframe without such rows
    '''
    # TODO: fiModelDescription/fiModelBase/fiModelSeries -- There are many where this info only appears once.
    # TODO: Unknown product size
    # TODO: Unknown Drive_System
    # TODO: Unknown Tire_Size
    # TODO: YearMade == 1000
    # TODO: _units contains "Unidentified"
    # Many of these todos cannot be todid, if we do in fact do them all, it leaves us without any data
    reasonable_to_prune = ['Hydraulics', 'Transmission', 'Enclosure', 'UsageBand', 'YearMade_categorical']
    for cname in reasonable_to_prune:
        if df[cname].dtype == object:
            df = df[df[cname] != 'Unknown']

    df = df[df['YearMade'] != 1000]
    return df

def remove_uninformative_columns(df):
    '''
        Given a dataframe assumed to have been loaded from our Tractor Data -- remove columns that don't provide
        much information. Specifically these are columns that have been determined to have a vast majority of
        "unknown" values in them, or columns that we determined experimentally do not contribute much to our
        regression models.

        returns DataFrame -- the input dataframe without such columns
    '''
    # Most of these are unknown on the vast majority of rows -- makes it tough to learn
    uninformative_cols = ['Forks', 'Pad_Type', 'Ride_Control', 'Turbocharged', 'Blade_Extension', 'Blade_Width', 'Enclosure_Type',
        'Engine_Horsepower', 'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Coupler', 'Coupler_System',
        'Grouser_Tracks', 'Hydraulics_Flow', 'Track_Type', 'Thumb', 'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting',
        'Blade_Type', 'Travel_Controls', 'Differential_Type', 'Steering_Controls', 'Undercarriage_Pad_Width'
    ]

    for cname in uninformative_cols:
        del df[cname]

    return df

def impute_median_all(df):
    '''
        Given a dataframe, find all the numeric types and impute the median values
        for any rows which are nan.

        return DataFrame -- the dataframe with all the nan values set to the imputed median
    '''
    for cname in df.columns:
        if df[cname].dtype != object:
            med_val = np.median(df[cname][~np.isnan(df[cname])])
            df[cname][np.isnan(df[cname])] = med_val

    return df

def dummify(df):
    '''
        Given a dataframe, for all the columns which are not numericly typed already,
        create dummies. This will NOT remove one of the dummies which is required for
        linear regression.

        returns DataFrame -- a dataframe with all non-numeric columns swapped into dummy columns
    '''
    obj_cols = []
    for cname in df.columns:
        if df[cname].dtype == object:
            obj_cols.append(cname)

    df = pd.get_dummies(df, columns=obj_cols)
    # for cname in obj_cols:
    #     del df[cname]

    return df


def get_multi_sale_tractors(df):
    '''
    return a dataframe that only has records for tractors whose machine id appears more than once

    '''
    machine_ids = df.MachineID
    cnt = machine_ids.value_counts()
    multi_sale = cnt[cnt > 1]
    multi_sale_tractors = df[df['MachineID'].isin(multi_sale.index)]
    return multi_sale_tractors


def merge_all_none_types(df):
    '''
    Using Configuration parameters above
     take all the
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
        return OUTPUT_UNKNOWN_VALUE
    elif value in INPUT_NONE_VALUES:
        return OUTPUT_UNKNOWN_VALUE
    elif value in KNOWN_INVALID_ENTRY_VALUES:
        return OUTPUT_UNKNOWN_VALUE

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
    if value in ['EROPS w AC', 'EROPS AC']:
        return 'Has AC'
    elif value in ['OROPS', 'OROPS', 'EROPS', 'NO ROPS']:
        return 'Likely No AC'
    else:
        return 'AC STATUS UNKNOWN'


def year_buckets_cat_map(val):
    '''
        Map a single year value to the 20 year buckets between 1940 and 2000,
        mark a known error (value 1000) as the configured OUTPUT_UNKNOWN_VALUE.
    '''
    if val == 1000:
        return OUTPUT_UNKNOWN_VALUE
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
    # TODO: These often fetch things like: Hydraulic Excavator, Track - Unidentified -- which
    # Have a direct mapping to a proper unit, but are not in the text.
    # Extract this mapping to make this more robust
    # It might be easier to do this simply using ProductGroupDesc
    result_name = re.search('[^0-9+]+$', desc_value)  # grab units from the tail..
    if result_name:
        units = result_name.group(0)
        return units
    else:
        return OUTPUT_UNKNOWN_VALUE


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
        result2 = re.search('([0-9\.]+)\s?\+', desc_value)
        if result2:
            value = result2.group(1)
            return float(value)
        else:
            return 0


def apply_all_specific_tractor_transforms(tractor_data):
    '''
        Performs several cleaning operations that merge specifically known problems in the
        data. See the applied functions. Operation is in place, returns the provided dataframe.
    '''
    tractor_data['Tire_Size'] = tractor_data['Tire_Size'].apply(combine_10_inch_tire_size)
    tractor_data['Modernity'] = tractor_data['YearMade'].apply(year_buckets_cat_map)
    tractor_data['AC_Status'] = tractor_data['Enclosure'].apply(enclosure_ac_map)
    tractor_data['_units'] = tractor_data['fiProductClassDesc'].apply(product_descr_to_units)
    tractor_data['_measurement'] = tractor_data['fiProductClassDesc'].apply(product_descr_to_mean)

    return tractor_data


### Aspirational Functions to Implement:
def add_sale_count(df):
    # TODO: Add column for "Sale Count" -- how many times had that tractor changed hands at for each
    # particular sale record.
    pass


def add_sale_year(df):
    # TODO: Parse saledate column to extract just the sale year
    pass
