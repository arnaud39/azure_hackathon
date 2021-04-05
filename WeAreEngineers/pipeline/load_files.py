import pandas as pd
import os


def load_data() -> dict:

    data = {}
    DIR = "WeAreEngineers/data"
    data["uhc_df"] = pd.read_csv(DIR + '/uhcCoverage.csv')
    data["cancer_df"] = pd.read_csv(DIR + '/30-70cancerChdEtc.csv')
    data["suicide_df"] = pd.read_csv(DIR + '/crudeSuicideRates.csv')
    data["sani_df"] = pd.read_csv(
        DIR + '/atLeastBasicSanitizationServices.csv')
    data["water_df"] = pd.read_csv(DIR + '/basicDrinkingWaterServices.csv')
    data["fueltech_df"] = pd.read_csv(DIR + '/cleanFuelAndTech.csv')
    data["gdi_df"] = pd.read_csv(DIR + '/gender_development.csv')
    data["genderequal_df"] = pd.read_csv(DIR + '/gender_inequality.csv')
    data["hdi_df"] = pd.read_csv(DIR + '/human_development.csv')
    return data
