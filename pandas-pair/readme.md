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
