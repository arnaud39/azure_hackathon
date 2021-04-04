import pandas as pd
from functools import reduce

def preprocess_data(
    data: dict
) -> dict:
    data["suicide_df"].rename(columns={
                              'First Tooltip': 'Crude suicide rates (per 100 000 population)'}, inplace=True)
    data["suicide_df"].drop(columns='Indicator', inplace=True)
    data["suicide_df"].drop(
        data["suicide_df"].loc[data["suicide_df"]['Dim1'] == 'Male'].index, inplace=True)
    data["suicide_df"].drop(
        data["suicide_df"].loc[data["suicide_df"]['Dim1'] == 'Female'].index, inplace=True)
    # process 30-70cancerChdEtc.csv
    data["cancer_df"].rename(columns={
                             'First Tooltip': 'Probability (%) of dying between age 30 and exact age 70 from any of cardiovascular disease, cancer, diabetes, or chronic respiratory disease'}, inplace=True)
    data["cancer_df"].drop(columns='Indicator', inplace=True)
    data["cancer_df"].drop(
        data["cancer_df"].loc[data["cancer_df"]['Dim1'] == 'Male'].index, inplace=True)
    data["cancer_df"].drop(
        data["cancer_df"].loc[data["cancer_df"]['Dim1'] == 'Female'].index, inplace=True)
    # process uhcCoverage.csv
    data["uhc_df"].rename(
        columns={'First Tooltip': 'UHC index of essential service coverage'}, inplace=True)
    data["uhc_df"].drop(columns='Indicator', inplace=True)
    # process atLeastBasicSanitizationServices.csv
    data["sani_df"].rename(columns={
                           'First Tooltip': 'Population using at least basic sanitation services (%)'}, inplace=True)
    data["sani_df"].drop(columns='Indicator', inplace=True)
    data["sani_df"].drop(data["sani_df"].loc[data["sani_df"]
                                             ['Dim1'] == 'Urban'].index, inplace=True)
    data["sani_df"].drop(data["sani_df"].loc[data["sani_df"]
                                             ['Dim1'] == 'Rural'].index, inplace=True)
    # process basicDrinkingWaterServices.csv
    data["water_df"].rename(columns={
                            'First Tooltip': 'Population using at least basic drinking-water services (%)'}, inplace=True)
    data["water_df"].drop(columns='Indicator', inplace=True)
    return data


def join_data(
    data: dict
) -> pd.DataFrame:
    data_frames = data.values()
    joined_df = reduce(lambda left, right: pd.merge(left, right, on=['Location', 'Period'],
                                             how='inner'), data_frames)
    joined_df.drop(columns={'Dim1_x', 'Dim1_y', 'Dim1'}, inplace=True)

    return joined_df

def set_index(
    data: pd.DataFrame
) -> pd.DataFrame:
    data_indexed = data.set_index("Location")
    return data_indexed
