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
