# WeAreEngineers

This repo contains a deep learning model to score the relevance of various socioeconomic dimensions against probability of different mortality causes. 

The socioeconomic factors we are looking at are:
1. Access to basic sanitation services
2. Access to basic drinking water services
3. Access to essential health services (based on UHC index of service coverage)

The mortality causes are:
1. Cardiovascular disease
2. Cancer
3. Diabetes
4. Chronic respiratory disease
5. Suicide

# Dataset
The individual dataset files used in this repo can be found in [`weAreEngineers/data`](https://github.com/azure-hack/azure_hackathon/tree/main/WeAreEngineers/data).
The dataset consists of health statistics of countries recognized by WHO and it was downloaded from [here](https://www.kaggle.com/utkarshxy/who-worldhealth-statistics-2020-complete?select=cleanFuelAndTech.csv).


# Setting Up
Run the following commands to install the required modules:
1. From the main project directory, `cd WeAreEngineers`
2. `pip3 install -r requirements.txt`

# How To Run
Run the following commands to execute the scripts:
1. In the main project directory, execute `python3 run.py`