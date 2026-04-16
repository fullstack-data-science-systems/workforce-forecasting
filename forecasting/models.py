"""Model architecture builders."""

from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, GlobalAveragePooling1D, GRU, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_lstm_model(input_shape, output_dim, learning_rate: float = 0.001):
    """Build Bidirectional LSTM model for multi-output forecasting."""
    model = Sequential(
        [
            Bidirectional(LSTM(128, return_sequences=True, activation="tanh"), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True, activation="tanh")),
            Dropout(0.2),
            LSTM(32, activation="tanh"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(output_dim, activation="linear"),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae", "mse"])
    return model


def build_gru_model(input_shape, output_dim, learning_rate: float = 0.001):
    """Build Bidirectional GRU model for multi-output forecasting."""
    model = Sequential(
        [
            Bidirectional(GRU(128, return_sequences=True, activation="tanh"), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(GRU(64, return_sequences=True, activation="tanh")),
            Dropout(0.2),
            GRU(32, activation="tanh"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(output_dim, activation="linear"),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae", "mse"])
    return model


def build_cnn_model(input_shape, output_dim, learning_rate: float = 0.001):
    """Build 1D CNN model for multi-output forecasting."""
    model = Sequential(
        [
            Conv1D(64, kernel_size=3, activation="relu", padding="causal", input_shape=input_shape),
            Conv1D(64, kernel_size=3, activation="relu", padding="causal"),
            Dropout(0.2),
            Conv1D(32, kernel_size=3, activation="relu", padding="causal"),
            GlobalAveragePooling1D(),
            Dense(64, activation="relu"),
            Dense(output_dim, activation="linear"),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae", "mse"])
    return model
