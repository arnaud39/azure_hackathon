from tensorflow.keras.losses import MAE
import numpy as np
def compute_score(model, y_true, X_test):
    y_pred = model.predict(X_test)
    mae = MAE(y_true, y_pred)
    return np.mean(mae.numpy())
