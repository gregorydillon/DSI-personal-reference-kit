import pandas as pd

HOSPITAL_DATA = pd.read_csv("data/hospital-costs.csv")


def expanded_financials(hospital_data):
    '''
        Return a copy of a provided DataFrame, hospital_data.
    '''
    clone = hospital_data.copy()
    clone['Total Charges'] = clone['Mean Charge'] * clone['Discharges']
    clone['Total Costs'] = clone['Mean Cost'] * clone['Discharges']
    clone['Markup'] = clone['Total Charges'] / clone['Total Costs']

    return clone


def discharges_by_description(hospital_data):
    '''
        Return a Series with the number of discharges grouped-and-summed by 'APR DRG Description'
        in the same order of the input data.
    '''
    return hospital_data.groupby('APR DRG Description').sum()['Discharges']


def sorted_by_profit(expanded_financials):
    '''
        Returns a reduced copy of the provided financial data that has been
        sorted by "Net Income" = Charges - Costs
    '''
    # 1. Create a new DataFrame named "net" that is only the
    # Facility Name, Total Charge, Total Cost from our original DataFrame
    net = expanded_financials.copy(['Facility Name', 'Total Charges', 'Total Costs'])


    # 2. Find the total amount each hospital spent, and how much they charged. (Group your
    # data by Facility names, and sum all the total costs and total charges)
    totals_by_hospital = net.groupby('Facility Name').sum()
    totals_by_hospital['Net Income'] = totals_by_hospital['Total Charges'] - totals_by_hospital['Total Costs']

    # 3. Now find the net income for every hospital.
    return totals_by_hospital.sort_values(by=['Net Income'])


def relevant_moderate_meningitis_stats(hospital_data):
    '''
        Given the hospital data, return a dataframe which contains only records
        for moderate cases of Viral Meningitis, and only for hospitals which have
        more than 3 discharges for Moderate Viral Meningitis. Additionally, reduce
        the dataframe to only columns:

        "Facility Name", "APR DRG Description","APR Severity of Illness Description","Discharges", "Mean Charge",
        "Median Charge", "Mean Cost"
    '''
    # Create a new dataframe that only contains the data corresponding to Viral Meningitis
    only_meningitis = hospital_data[hospital_data["APR DRG Description"] == "Viral Meningitis"]

    # Now, with our new dataframe, only keep the data columns we care about which are:
    relevant_columns = ["Facility Name", "APR DRG Description", "APR Severity of Illness Description",
                        "Discharges", "Mean Charge", "Median Charge", "Mean Cost"]

    reduced_meningitis = only_meningitis[relevant_columns]

    # Find which hospital is the least expensive (based on "Mean Charge") for treating Moderate cases of VM.
    relevant_moderate_meningitis = reduced_meningitis[
        (reduced_meningitis["APR Severity of Illness Description"] == "Moderate") &
        (reduced_meningitis["Discharges"] > 3)
    ]

    return relevant_moderate_meningitis.sort_values(by=['Mean Charge'])


def severity_cost_correlation(hospital_data):
    severity_values = {
        'Minor': 0,
        'Moderate': 1,
        'Major': 2,
        'Extreme': 3
    }

    # Give unforseen values an 'other' category.
    def map_severity(severity):
        return severity_values.get(severity, 4)

    clone = hospital_data.copy()
    clone['APR Severity of Illness Description'] = clone['APR Severity of Illness Description'].apply(map_severity)
    correlation_strength = clone['APR Severity of Illness Description'].corr(clone['Mean Charge'])

    return correlation_strength


def discharges_by_disease_and_severity(hospital_data):
    '''
    Return a dataframe grouped by disease and severity, with sums in all columns.
    Returned dataframe has cols: "APR DRG Description", 'APR Severity of Illness Description',
                                 "Discharges", "Mean Cost"
    '''
    # First reduce to the columns we care about
    discharge_severity_cost = hospital_data[
        ["APR DRG Description", 'APR Severity of Illness Description', "Discharges", "Mean Cost"]
    ]

    discharges_cost_by_disease_severity = discharge_severity_cost.groupby(
        ["APR DRG Description", 'APR Severity of Illness Description']
    ).sum()

    return discharges_cost_by_disease_severity
