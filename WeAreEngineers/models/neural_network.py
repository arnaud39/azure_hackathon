"""Keras model."""
from .base_models import KerasModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping


def get_model():
    """Return a random forest regressor model."""
    # Define dense architecture
    neural_network = Sequential()
    for _ in range(4):
        neural_network.add(Dense(
            30,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=8e-2, l2=8e-2),
            bias_regularizer=regularizers.l2(8e-2),
            activity_regularizer=regularizers.l2(8e-2)
        ))
    neural_network.add(Dense(1))
    neural_network.compile(loss="mae", optimizer=Adam(lr=0.0005166664147198357), metrics=["mae"])

    model = KerasModel(
        model=neural_network,
        name="neural_network",
        epochs = 5000,
        batch_size = 16,
        verbose = 1,
        callbacks = EarlyStopping(
            patience=30, monitor="val_loss", min_delta=5e-5, restore_best_weights=True
        )
    )

    return model
