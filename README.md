# WeAreEngineers

This repo contains a deep learning model to score the relevance of various socioeconomic dimensions against probability of different mortality causes. 

The socioeconomic factors we are looking at are:
1. Access to basic sanitation services
2. Access to basic drinking water services
3. Access to essential health services (based on UHC index of service coverage)
4. Key dimensions of human development: a long and healthy life, access to knowledge, and a decent standard of living (encapsulated as Human Development Index)
5. Male/Female life expectancy at birth, years of childhood education and and mean years of education over 25 years old (encapsulated as Gender Development Index)
6. Gender-based disadvantage in reproductive health, empowerment and the labour market (encapsulated as Gender Inequality Index)
7. Crude suicide rates

The mortality causes are:
1. Cardiovascular disease
2. Cancer
3. Diabetes
4. Chronic respiratory disease

# Dataset
The individual dataset files used in this repo can be found in [`weAreEngineers/data`](https://github.com/azure-hack/azure_hackathon/tree/main/WeAreEngineers/data).
<br>The dataset consists of health statistics of countries recognized by WHO and it was downloaded from [here](https://www.kaggle.com/utkarshxy/who-worldhealth-statistics-2020-complete?select=cleanFuelAndTech.csv).
<br>The dataset consists of index scores (Human Development Inde, Gender Development Index and Gender Inequality Index) of countries recognized by WHO and it was downloaded from [here](https://www.kaggle.com/undp/human-development?select=multidimensional_poverty.csv).

# Setting Up
From the main project directory, `pip install -r WeAreEngineers/requirements.txt`

# How To Run
From the main project directory, execute `python3 run.py`