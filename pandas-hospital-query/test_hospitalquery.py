import pandas as pd
import hospitalquery as hq


# Please don't modify!
HOSPITAL_DATA = pd.read_csv("data/hospital-costs.csv")

def test_expanded_financials():
    expanded = hq.expanded_financials(HOSPITAL_DATA)
    assert expanded is not HOSPITAL_DATA

    # Total Charges: Mean Charges * Discharges
    assert expanded['Total Costs'].loc[0] == 588240.0

    # Total Costs: Mean Charges * Discharges
    assert expanded['Total Charges'].loc[0] == 1083867.0

    # Markup: Total Charges / Total Costs
    assert expanded['Markup'].loc[0] == 1.8425591595267239


def test_discharges_by_description():
    sdd = hq.discharges_by_description(HOSPITAL_DATA)
    # just a few
    assert sdd['Pancreas Transplant'] == 120
    assert sdd['Neonate W Ecmo'] == 123
    assert sdd['Principal Diagnosis Invalid As Discharge Diagnosis'] == 129
    assert sdd['Extensive 3Rd Degree Burns W Skin Graft'] == 132
    assert sdd['Neonate, Transferred <5 Days Old, Not Born Here'] == 224
    assert sdd['Ungroupable'] == 359
    assert sdd['Neonate Birthwt 1000-1249G W Or W/O Other Significant Condition'] == 477
    assert sdd['Neonatal Aftercare'] == 497
    assert sdd['Craniotomy For Multiple Significant Trauma'] == 515
    assert sdd['Mental Illness Diagnosis W O.R. Procedure'] == 532
    assert sdd['Neonate Bwt 1500-2499G W Major Procedure'] == 613
    assert sdd['Heart &/Or Lung Transplant'] == 616
    assert sdd['Neonate Birthwt >2499G W Major Cardiovascular Procedure'] == 679
    assert sdd['Neonate Birthwt 1500-1999G W Congenital/Perinatal Infection'] == 702
    assert sdd['Extensive 3Rd Degree Or Full Thickness Burns W/O Skin Graft'] == 783
    assert sdd['Neonate Bwt <1500G W Major Procedure'] == 845
    assert sdd['Neonate Bwt <500G Or Ga <24 Weeks'] == 960
    assert sdd['Major Larynx & Trachea Procedures'] == 987
    assert sdd['Neonate Birthwt >2499G W Other Major Procedure'] == 1009
    assert sdd['Radiotherapy'] == 1069
    assert sdd['Neonate Bwt 1250-1499G W Or W/O Other Significant Condition'] == 1120


def test_sorted_by_profit():
    expanded_financials = hq.expanded_financials(HOSPITAL_DATA)
    sorted_by_profit = hq.sorted_by_profit(expanded_financials)

    assert sorted_by_profit.iloc[0].loc['Net Income'] == -194816068.0
    assert sorted_by_profit.iloc[0].name == 'TLC Health Network Tri-County Memorial Hospital'

    assert sorted_by_profit.iloc[-1].loc['Net Income'] == 6050732279.0
    assert sorted_by_profit.iloc[-1].name == 'North Shore University Hospital'


def test_relevant_moderate_meningitis():
    relevant_moderate_meningitis = hq.relevant_moderate_meningitis_stats(HOSPITAL_DATA)
    assert relevant_moderate_meningitis.iloc[0].loc['Facility Name'] == 'Cayuga Medical Center at Ithaca'
    assert relevant_moderate_meningitis.iloc[0].loc['Discharges'] == 6
    assert relevant_moderate_meningitis.iloc[0].loc['Mean Charge'] == 5738

    assert relevant_moderate_meningitis.iloc[-1].loc['Facility Name'] == 'St Lukes Roosevelt Hospital - St Lukes Hospital Division'
    assert relevant_moderate_meningitis.iloc[-1].loc['Discharges'] == 4
    assert relevant_moderate_meningitis.iloc[-1].loc['Mean Charge'] == 79245


def test_severity_cost_correlation():
    assert hq.severity_cost_correlation(HOSPITAL_DATA) == 0.36647546172361339


def test_discharges_by_disease_and_severity():
    discharges_cost_by_disease_severity = hq.discharges_by_disease_and_severity(HOSPITAL_DATA)

    most_discharged = discharges_cost_by_disease_severity.sort_values(by=["Discharges"])
    assert most_discharged.iloc[-1].loc['Discharges'] == 528921.0
    assert most_discharged.iloc[-1].loc['Mean Cost'] == 670070.0
    assert most_discharged.iloc[-1].name == ('Neonate Birthwt >2499G, Normal Newborn Or Neonate W Other Problem', 'Minor')

    most_expensive = discharges_cost_by_disease_severity.sort_values(by=["Mean Cost"])
    assert most_expensive.iloc[-1].loc['Discharges'] == 8546.0
    assert most_expensive.iloc[-1].loc['Mean Cost'] == 69085412.0
    assert most_expensive.iloc[-1].name == ('Tracheostomy W MV 96+ Hours W Extensive Procedure Or Ecmo', 'Extreme')
