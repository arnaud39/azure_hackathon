from .pipeline.load_files import load_data
from .pipeline.preprocess import preprocess_data, join_data, set_index
from .pipeline.config_utils import split_x_y, split_train_test_valid, scale_data
from .models.compute_score import compute_score
from .models.base_models import KerasModel
import eli5
from eli5.sklearn import PermutationImportance
import logging
import os
import importlib
from pyfiglet import Figlet
from IPython.core.display import display, HTML
import numpy as np

logger = logging.getLogger(__name__)

def run(model_name: str):


    if logger.getEffectiveLevel() <= logging.INFO:
        os.system("cls" if os.name == "nt" else "clear")
    logger.info(Figlet(font="slant").renderText("Azure Hackathon"))

    data = preprocess_data(load_data())
    data_joined = join_data(data)
    data_indexed = set_index(data_joined)
    logger.info(data_indexed.columns)

    X, y = split_x_y(data_indexed)
    ((X_train_raw, y_train), (X_validate_raw, y_validate),
     (X_test_raw, y_test)) = split_train_test_valid(X, y)

    X_train, scaler = scale_data(X_train_raw)
    X_validate, _ = scale_data(X_validate_raw, scaler)
    X_test, _ = scale_data(X_test_raw, scaler)

    training_size_x = X_train.shape
    testing_size_x = X_test.shape
    validation_size_x = X_validate.shape
    training_size_y = y_train.shape
    testing_size_y = y_test.shape
    validation_size_y = y_validate.shape

    logger.info("\n\n" + "=" * 20)
    logger.info("Data is ready, let's train ! 😉")
    logger.info("Training :")
    logger.info(f"  X size: {training_size_x}")
    logger.info(f"  y size: {training_size_y}")
    logger.info("Validation :")
    logger.info(f"  X size: {validation_size_x}")
    logger.info(f"  y size: {validation_size_y}")
    logger.info("Testing :")
    logger.info(f"  X size: {testing_size_x}")
    logger.info(f"  y size: {testing_size_y}")
    logger.info(X_train_raw.columns)
    logger.info(y_train.columns)

    try:
        model = importlib.import_module(
            name="WeAreEngineers.models." + model_name
        ).get_model()
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"The model {model_name} doesn't exist") from e

    if "search" in model_name:
        optimum = model.best_params()
        file = open("best_params.txt", "a")
        file.write(model_name + "\n" + str(optimum))
        file.close()
        model.fit(X_train,y_train, validation_data=(np.append(X_validate, X_test), np.append(y_validate,y_test)))

    elif isinstance(model, KerasModel):
        model.fit(X_train,y_train, validation_data=(X_validate, y_validate))
    else:
        model.fit(np.append(X_train, X_validate, axis=0),np.append(y_train,y_validate))
    logger.info("MAE:")
    logger.info(compute_score(model, y_test, X_test))

    perm = PermutationImportance(model, random_state=1, scoring="neg_mean_absolute_error").fit(X_train,y_train)
    print(eli5.format_as_dataframe(eli5.explain_weights(perm, feature_names = X_train_raw.columns.tolist())))
