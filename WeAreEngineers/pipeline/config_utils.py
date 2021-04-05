import pandas as pd
import sklearn.preprocessing
from typing import Tuple, Any
import numpy as np


def split_x_y(
    data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_x = data.drop(
        columns='Probability (%) of dying between age 30 and exact age 70 from any of cardiovascular disease, cancer, diabetes, or chronic respiratory disease', axis=0)
    df_y = data[[
        'Probability (%) of dying between age 30 and exact age 70 from any of cardiovascular disease, cancer, diabetes, or chronic respiratory disease']]
    return (df_x, df_y)


def split_train_test_valid(
    X: pd.DataFrame,
    y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_part = 0.67
    valid_part = 0.13
    test_part = 0.2

    train, validate, test = zip(
        np.split(
            X,
            [int(train_part * len(X)), int((train_part + valid_part) * len(X))],
            axis=0,
        ),
        np.split(
            y,
            [int(train_part * len(y)), int((train_part + valid_part) * len(y))],
            axis=0,
        )
    )

    return (train, validate, test)


def scale_data(
    data: pd.DataFrame,
    scaler: Any = None
) -> Tuple[pd.DataFrame, Any]:
    if scaler is None:
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(data)
    data_scaled = scaler.transform(data)

    return (data_scaled, scaler)
