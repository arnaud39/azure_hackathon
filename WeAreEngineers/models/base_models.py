"""Definition of the base classes for all machine learning models."""
from tensorflow.keras.models import load_model
from numpy.random import seed
import os
from glob import glob
import joblib
# Shut up, tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class SkModel:
    """Base class for sklearn models."""

    def __init__(self, name, model):
        """Init method."""
        self.model = model
        self.name = name
        seed(1)

    def save(self, folder):
        """Save the model parameters in the given folder."""
        joblib.dump(self.model, os.path.join(folder, self.name + ".sklearn"))

    def load(self, folder):
        """Load the model parameters from the given folder."""
        files = glob(os.path.join(folder, "*.sklearn"))
        if len(files) != 1:
            raise ValueError(
                f"Couldn't load model, {len(files)} "
                f".sklearn files found in {folder} instead of the expected 1."
            )
        self.model = joblib.load(files[0])

    def fit(self, *args, **kwargs):
        """Train the model."""
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Use the model."""
        return self.model.predict(*args, **kwargs)


class KerasModel:
    """Base class for keras models."""

    def __init__(self, name, model, **kwargs):
        """Init method."""
        seed(1)
        self.model = model
        self.name = name
        self.training_kwargs = kwargs

    def save(self, folder):
        """Save the model parameters in the given folder."""
        self.model.save(os.path.join(folder, self.name + ".keras"))

    def load(self, folder):
        """Load the model parameters from the given folder."""
        files = glob(os.path.join(folder, "*.keras"))
        if len(files) != 1:
            raise ValueError(
                f"Couldn't load model, {len(files)} "
                f".keras files found in {folder} instead of the expected 1."
            )
        self.model = load_model(files[0])

    def fit(self, *args, **kwargs):
        """Train the model."""
        return self.model.fit(
            use_multiprocessing=True, *args, **kwargs, **self.training_kwargs
        )

    def predict(self, *args, **kwargs):
        """Use the model."""
        pred = self.model.predict(*args, **kwargs)
        # Flatten the prediction if it's a column vector for coherence with sklearn
        if pred.shape[1] == 1:
            return pred.flatten()
        return pred

    def best_params(self):
        """Use for randomized search."""
        return self.model.best_params_
