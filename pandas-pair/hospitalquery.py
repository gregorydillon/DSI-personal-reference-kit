import pandas as pd
HOSPITAL_DATA = pd.read_csv("data/hospital-costs.csv")

def expanded_financials(hospital_data):

    clone = hospital_data.copy()
    clone['Total Charges'] = clone['Mean Charge'] * clone['Discharges']
    clone['Total Costs'] = clone['Mean Cost'] * clone['Discharges']
    clone['Markup'] = clone['Total Charges'] / clone['Total Costs']

    return clone

def discharges_by_description(hospital_data):
    return hospital_data.groupby('APR DRG Description').sum()['Discharges']


def sorted_by_profit(expanded_financials):
    # 1. Create a new DataFrame named "net" that is only the
    # Facility Name, Total Charge, Total Cost from our original DataFrame
    net = pd.DataFrame({
        'Facility Name': expanded_financials['Facility Name'],
        'Total Charges': expanded_financials['Total Charges'],
        'Total Costs': expanded_financials['Total Costs']
    })

    # 2. Find the total amount each hospital spent, and how much they charged. (Group your
    # data by Facility names, and sum all the total costs and total charges)
    totals_by_hospital = net.groupby('Facility Name').sum()
    totals_by_hospital['Net Income'] =  totals_by_hospital['Total Charges'] - totals_by_hospital['Total Costs']

    # 3. Now find the net income for every hospital.
    return totals_by_hospital.sort_values(by=['Net Income'])


def relevant_moderate_meningitis_stats(hospital_data):
    '''
        Given the hospital data, return a dataframe which contains only records
        for moderate cases of Viral Meningitis, and only for hospitals which have
        more than 3 discharges for Moderate Viral Meningitis. Additionally, reduce
        the dataframe to only columns:

        "Facility Name", "APR DRG Description","APR Severity of Illness Description","Discharges", "Mean Charge", "Median Charge", "Mean Cost"
    '''
    # Create a new dataframe that only contains the data corresponding to Viral Meningitis
    only_meningitis = hospital_data.loc[hospital_data["APR DRG Description"] == "Viral Meningitis"]

    # Now, with our new dataframe, only keep the data columns we care about which are:
    relevant_columns = ["Facility Name", "APR DRG Description","APR Severity of Illness Description",
                        "Discharges", "Mean Charge", "Median Charge", "Mean Cost"]
    reduced_meningitis = only_meningitis[relevant_columns].copy()

    # Find which hospital is the least expensive (based on "Mean Charge") for treating Moderate cases of VM.
    moderate_meningitis = reduced_meningitis.loc[
        reduced_meningitis["APR Severity of Illness Description"] == "Moderate"
    ]

    relevant_moderate_meningitis = moderate_meningitis.loc[
        reduced_meningitis["Discharges"] > 3
    ]

    return relevant_moderate_meningitis.sort_values(by=['Mean Charge'])


def severity_cost_correlation(hospital_data):
    clone = hospital_data.copy()

    # To do this I'm going to give the categories numerical values
    def transform_severity(severity):
        if severity == 'Minor':
            return 0
        elif severity == 'Moderate':
            return 1
        elif severity == 'Major':
            return 2
        elif severity == 'Extreme':
            return 3
        else:
            return 4 # Some kinda ... unknown severity?

    clone['APR Severity of Illness Description'] = clone['APR Severity of Illness Description'].apply(transform_severity)
    correlation_strength = clone['APR Severity of Illness Description'].corr(clone['Mean Charge'])

    return correlation_strength


def discharges_by_disease_and_severity(hospital_data):
    # First reduce to the columns we care about
    discharge_severity_cost = hospital_data[
        ["APR DRG Description", 'APR Severity of Illness Description', "Discharges", "Mean Cost"]
    ].copy()

    discharges_cost_by_disease_severity = discharge_severity_cost.groupby(
        ["APR DRG Description", 'APR Severity of Illness Description']
    ).sum()

    return discharges_cost_by_disease_severity
