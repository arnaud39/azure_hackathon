"""Random forest model."""
from sklearn.ensemble import RandomForestRegressor
from .base_models import SkModel


def get_model():
    """Return a random forest regressor model."""
    model = SkModel(model=RandomForestRegressor(), name="random_forest")
    return model
