## Setup

```
pip install pandas
pip install pytest
```

##### Running The Tests

```
pytest
```

## Using Pandas:

Pandas is a powerful tool for examining and manipulating data. You can:

> For more context on these examples, dive into the source.

* Create columns using other columns; perform operations over those columns:
  ```python
  df = hospital_data.copy()
  df['Total Charges'] = df['Mean Charge'] * df['Discharges']
  df['Total Costs'] = df['Mean Cost'] * df['Discharges']
  df['Markup'] = df['Total Charges'] / df['Total Costs']

  return df
  ```

* Group rows using aggregate functions (just like SQL's `GROUP BY`):
  ```python
    discharges_by_disease = hospital_data.groupby('Disease Name').sum()
  ```

* Sort things:
  ```python
  totals_by_hospital.sort_values(by=['Net Income'])
  ```

* Select data over filtered across many rows (parallel DataFrames or Series as well):
```python
relevant_moderate_meningitis = reduced_meningitis[
    (reduced_meningitis["APR Severity of Illness Description"] == "Moderate") &
    (reduced_meningitis["Discharges"] > 3)
]
```

* Apply arbitrary transformation functions to columns using a functional style!
  ```python
  map_rules = {
      'Minor': 0,
      'Moderate': 1,
      'Major': 2,
      'Extreme': 3
  }

  # Give unforseen values an 'other' category.
  def map_fn:
      return map_rules.get(x, 4)

  df['Severity'] = df['Severity'].apply(map_fn)
  ```
