"""Prediction and forecasting helpers."""

import numpy as np


def make_predictions(models_dict, X_train, X_val, X_test):
    """Make predictions for all models."""
    predictions = {}

    for model_name, model in models_dict.items():
        predictions[model_name] = {
            "train": model.predict(X_train, verbose=0),
            "val": model.predict(X_val, verbose=0),
            "test": model.predict(X_test, verbose=0),
        }

    return predictions


def inverse_transform_predictions(predictions, scaler):
    """Inverse transform scaled predictions back to original scale."""
    return scaler.inverse_transform(predictions)


def forecast_future(model, last_sequence, n_steps: int, scaler):
    """Forecast future values using iterative prediction."""
    forecasts = []
    current_sequence = last_sequence.copy()

    for _ in range(n_steps):
        input_seq = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        next_pred = model.predict(input_seq, verbose=0)[0]
        forecasts.append(next_pred)
        current_sequence = np.vstack([current_sequence[1:], next_pred])

    forecasts = np.array(forecasts)
    return scaler.inverse_transform(forecasts)
