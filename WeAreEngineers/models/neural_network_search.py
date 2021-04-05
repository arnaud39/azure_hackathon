"""Keras model."""
from .base_models import KerasModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV



def get_model():
    """Return a neural network model."""
    param_distribs = {
        "n_hidden": range(2, 10),
        "n_neurons": list(range(5, 100)),
        "learning_rate": reciprocal(3e-4, 3e-3),
        "activation": ["relu","sigmoid","tanh"],
    }

    def build_model(
        n_hidden: int = 1,
        n_neurons: int = 32,
        learning_rate: float = 9e-3,
        activation: str = "relu"
    ):
        model = Sequential()
        for _ in range(n_hidden):
            model.add(Dense(n_neurons,
                            activation=activation,
                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
                            bias_regularizer=regularizers.l2(1e-5),
                            activity_regularizer=regularizers.l2(1e-5)))
        model.add(Dense(1, activation="relu"))
        model.compile(loss="mae", optimizer=Adam(
            lr=learning_rate), metrics=["mae"])
        return model

    keras_reg = KerasRegressor(build_model)
    search = RandomizedSearchCV(keras_reg,
                                param_distribs,
                                n_iter=200,
                                cv=2,
                                scoring="neg_mean_absolute_error")

    model = KerasModel(
        model=search,
        name="neural_network",
        epochs=500,
        batch_size=32,
        callbacks=EarlyStopping(
            patience=30, monitor="val_loss", min_delta=5e-5, restore_best_weights=True
        )
    )

    return model
