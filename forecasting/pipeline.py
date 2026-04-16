"""Main execution pipeline for employment forecasting."""

import pickle

import pandas as pd

from .data import (
    create_sequences,
    explore_dataset,
    handle_missing_values,
    load_data,
    pivot_employment_data,
    scale_features,
    split_data,
)
from .evaluation import calculate_metrics
from .prediction import forecast_future
from .runtime import configure_runtime, print_runtime_info


def run_pipeline(data_path: str = "example_data.csv") -> None:
    """Run the full deep-learning time-series forecasting pipeline."""
    configure_runtime()

    # Import TensorFlow-dependent modules only after runtime flags are configured.
    from .models import build_cnn_model, build_gru_model, build_lstm_model
    from .training import train_model

    import tensorflow as tf
    from tensorflow import keras

    print_runtime_info(tf, keras)

    print("=" * 70)
    print("DEEP LEARNING TIME-SERIES FORECASTING: CANADIAN EMPLOYMENT RATES")
    print("=" * 70)

    print("\n[1/10] Loading data...")
    df_original = load_data(data_path)

    print("\n[2/10] Exploring dataset...")
    explore_dataset(df_original)

    print("\n[3/10] Pivoting data...")
    df_pivot = pivot_employment_data(df_original)
    df_pivot.to_csv("df_pivot.csv", index=False)
    print(f"Pivoted dataset shape: {df_pivot.shape}")

    print("\n[4/10] Preprocessing data...")
    df_pivot["month"] = pd.to_datetime(df_pivot["month"])
    df_clean, numeric_cols = handle_missing_values(df_pivot)

    dates = df_clean["month"].values
    features = df_clean[numeric_cols].values

    scaled_features, scaler = scale_features(features)

    print("\n[5/10] Creating sequences...")
    n_steps = 12
    X, y = create_sequences(scaled_features, n_steps)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    print("\n[6/10] Building models...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_dim = y_train.shape[1]

    model_lstm = build_lstm_model(input_shape, output_dim)
    model_gru = build_gru_model(input_shape, output_dim)
    model_cnn = build_cnn_model(input_shape, output_dim)

    print("Models created successfully")

    print("\n[7/10] Training models...")
    print("Training LSTM...")
    history_lstm = train_model(model_lstm, X_train, y_train, X_val, y_val)

    print("\nTraining GRU...")
    history_gru = train_model(model_gru, X_train, y_train, X_val, y_val)

    print("\nTraining CNN...")
    history_cnn = train_model(model_cnn, X_train, y_train, X_val, y_val)

    print("\n[8/10] Evaluating models...")
    models_dict = {"LSTM": model_lstm, "GRU": model_gru, "CNN": model_cnn}

    for model_name, model in models_dict.items():
        y_test_pred = model.predict(X_test, verbose=0)
        calculate_metrics(y_test, y_test_pred, f"Test ({model_name})")

    print("\n[9/10] Generating forecasts...")
    last_known_sequence = scaled_features[-n_steps:]

    future_forecasts_lstm = forecast_future(model_lstm, last_known_sequence, 60, scaler)
    future_forecasts_gru = forecast_future(model_gru, last_known_sequence, 60, scaler)
    future_forecasts_cnn = forecast_future(model_cnn, last_known_sequence, 60, scaler)

    last_date = pd.to_datetime(dates[-1])
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=60, freq="MS")

    forecast_df_lstm = pd.DataFrame(future_forecasts_lstm, columns=numeric_cols)
    forecast_df_lstm.insert(0, "month", forecast_dates)

    forecast_df_gru = pd.DataFrame(future_forecasts_gru, columns=numeric_cols)
    forecast_df_gru.insert(0, "month", forecast_dates)

    forecast_df_cnn = pd.DataFrame(future_forecasts_cnn, columns=numeric_cols)
    forecast_df_cnn.insert(0, "month", forecast_dates)

    print("\n[10/10] Saving results...")
    forecast_df_lstm.to_csv("employment_forecasts_2020_2025_lstm.csv", index=False)
    forecast_df_gru.to_csv("employment_forecasts_2020_2025_gru.csv", index=False)
    forecast_df_cnn.to_csv("employment_forecasts_2020_2025_cnn.csv", index=False)

    model_lstm.save("employment_forecast_lstm_final.h5")
    model_gru.save("employment_forecast_gru_final.h5")
    model_cnn.save("employment_forecast_cnn_final.h5")

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\n" + "=" * 70)
    print("PROJECT COMPLETE - ALL DELIVERABLES GENERATED")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  1. df_pivot.csv - Pivoted dataset")
    print("  2. employment_forecasts_2020_2025_lstm.csv - LSTM forecasts")
    print("  3. employment_forecasts_2020_2025_gru.csv - GRU forecasts")
    print("  4. employment_forecasts_2020_2025_cnn.csv - CNN forecasts")
    print("  5. employment_forecast_lstm_final.h5 - LSTM model")
    print("  6. employment_forecast_gru_final.h5 - GRU model")
    print("  7. employment_forecast_cnn_final.h5 - CNN model")
    print("  8. scaler.pkl - Feature scaler")
    print("\nExecution completed successfully!")
