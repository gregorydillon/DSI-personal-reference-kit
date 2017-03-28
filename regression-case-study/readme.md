# Tractor Pricing

This subdirectory contains a version of the first case study. The code must be run from this directory for the module systems to work. You can examine the code in `validate.py`'s `main` function to get an idea of how these python modules work together to clean, and validate data. To run that first unzip the training data:

```
unzip data/Train.zip
```

Then run the validator main:

```
python validate.py
```

There is unfortnuately no cross validation as of yet, just a random sample of test data left out each time you test a formula. For trying many formulas I suggest using `validate_interactive.py` in order to keep cleaned data in memory instead of loading it up every time (which happens if you run the main in `validate.py`):

```
ipython -i validate_interactive

# NOW PASTE INTO PROMPT:
model, score = test_formula(favorite_so_far, tractor_data, col_to_predict
print score
```

Prewrite a bunch of formulas, or just iterate on a few and cll test_formula. It's not so slow once the data is loaded and cleaned.


### Coolest Things In This Repo:

__Data Cleanup__: The input data is pretty messy. We've done a lot of transformations in cleaner to try and make the data more consistent. We've added a few additional columns as well. Much of this is using the 'apply' method and passing functions:

```python
tractor_data['Tire_Size'] = tractor_data['Tire_Size'].apply(combine_10_inch_tire_size)
tractor_data['Modernity'] = tractor_data['YearMade'].apply(year_buckets_cat_map)
tractor_data['AC Status'] = tractor_data['Enclosure'].apply(enclosure_ac_map)
tractor_data['_units'] = tractor_data['fiProductClassDesc'].apply(product_descr_to_units)
tractor_data['_measurement'] = tractor_data['fiProductClassDesc'].apply(product_descr_to_mean)
```

A look at one of the functions passed to apply, which predicts if the vehicle has AC or not based on a specific columns known values:

```python
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
```

This function runs for every value in the series on which it is called, and the result for the new column being made is the returned value.

Another uses regex to extract values from text that is consistently formatted:

```python
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
```

__Interaction Terms__: We're using the StatsModels formula API which makes it easy to create interaction terms. This is done in `validate.py`'s main:

```python
col_to_predict = 'SalePrice'
favorite_so_far = '''
  SalePrice ~ YearMade +
          Modernity +
          MachineHoursCurrentMeter +
          ProductGroup +
          Enclosure +
          Hydraulics +
          _units*_measurement  # INTERACTION TERM -- consider each units/measurements combo individually
  '''

  model, score = test_formula(favorite_so_far, tractor_data, col_to_predict)
  print "LRMSE: {}".format(score)
```

This lets the algorithm consider two features in conjunction rather than just as a linear combination.
