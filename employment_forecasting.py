"""
Deep Learning Time-Series Forecasting: Employment Rate in Canada (2020-2025)

Project: Predicting Employment Rates across Canadian Provinces using LSTM/GRU/1D-CNN Neural Networks
Dataset: Employment Rate in Canada (https://www.kaggle.com/datasets/ortizmacleod/employment-rate-canada)

This script develops deep learning models for forecasting employment rates across multiple Canadian provinces
for the period 2020-2025, predicting four key metrics per province:
- Full-time Male employment
- Full-time Female employment
- Part-time Male employment
- Part-time Female employment
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import os
import warnings
from datetime import datetime

# Configure TensorFlow runtime flags before importing tensorflow.
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, Bidirectional, 
                                     Concatenate, Conv1D, GlobalAveragePooling1D, 
                                     MaxPooling1D, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")


# ============================================================================
# SECTION 2: DATA LOADING AND EXPLORATION
# ============================================================================

def load_data(filepath):
    """Load the employment data from CSV"""
    df = pd.read_csv(filepath)
    print("=== Original Dataset Shape ===")
    print(f"Shape: {df.shape}")
    print(f"First 10 rows:")
    print(df.head(10))
    return df


def explore_dataset(df_original):
    """Explore the dataset structure"""
    print("=== Dataset Information ===")
    print(df_original.info())
    print("\n=== Statistical Summary ===")
    print(df_original.describe())
    print("\n=== Unique Values ===")
    print(f"Variables: {df_original['variable'].unique()}")
    print(f"Sex Categories: {df_original['sex'].unique()}")
    print(f"Provinces: {df_original.columns[3:].tolist()}")
    print(f"Date Range: {df_original['month'].min()} to {df_original['month'].max()}")
    print(f"Total Records: {len(df_original)}")


# ============================================================================
# SECTION 3: DATA PIVOTING & TRANSFORMATION
# ============================================================================

def pivot_employment_data(df):
    """
    Transform employment data from long to wide format.
    
    Args:
        df: Original dataframe with month, variable, sex, and province columns
    
    Returns:
        Pivoted dataframe with Province_Type_Gender columns
    """
    df_pivot_list = []
    provinces = df.columns[3:].tolist()

    df_ft = df[df['variable'] == 'Full-time employment'].copy()
    df_pt = df[df['variable'].str.strip() == 'Part-time employment'].copy()

    # Process each province
    for province in provinces:
        # Full-time Female
        ft_f = df_ft[df_ft['sex'] == 'Females'][['month', province]].copy()
        ft_f.columns = ['month', f'{province}_FT_F']

        # Full-time Male
        ft_m = df_ft[df_ft['sex'] == 'Males'][['month', province]].copy()
        ft_m.columns = ['month', f'{province}_FT_M']

        # Part-time Female
        pt_f = df_pt[df_pt['sex'] == 'Females'][['month', province]].copy()
        pt_f.columns = ['month', f'{province}_PT_F']

        # Part-time Male
        pt_m = df_pt[df_pt['sex'] == 'Males'][['month', province]].copy()
        pt_m.columns = ['month', f'{province}_PT_M']

        df_pivot_list.extend([ft_f, ft_m, pt_f, pt_m])

    # Merge all dataframes on month
    df_result = df_pivot_list[0]
    for df_temp in df_pivot_list[1:]:
        df_result = df_result.merge(df_temp, on='month', how='outer')

    # Sort by month and reset index
    df_result = df_result.sort_values('month').reset_index(drop=True)
    return df_result


# ============================================================================
# SECTION 4: DATA PREPROCESSING
# ============================================================================

def handle_missing_values(df):
    """Handle missing values using forward and backward fill"""
    print("=== Handling Missing Values ===")
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    # Forward fill then backward fill
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(method='bfill')

    remaining_missing = df_clean[numeric_cols].isnull().sum().sum()
    print(f"Remaining missing values: {remaining_missing}")
    return df_clean, numeric_cols


def scale_features(features):
    """Scale features using MinMaxScaler"""
    print("=== Feature Scaling ===")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    print(f"Scaled data shape: {scaled_features.shape}")
    print(f"Scaled data range: [{scaled_features.min():.4f}, {scaled_features.max():.4f}]")
    return scaled_features, scaler


def create_sequences(data, n_steps):
    """
    Create sequences for time series prediction.
    
    Args:
        data: Scaled feature array
        n_steps: Number of time steps to look back
    
    Returns:
        X: Input sequences (samples, time_steps, features)
        y: Target values (samples, features)
    """
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def split_data(X, y, train_ratio=0.70, val_ratio=0.15):
    """Split data into train, validation, and test sets"""
    print("=== Data Split ===")
    
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    test_size = len(X) - train_size - val_size

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"Training set: {X_train.shape[0]} samples ({train_size/len(X)*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({val_size/len(X)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({test_size/len(X)*100:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================================
# SECTION 5: MODEL ARCHITECTURES
# ============================================================================

def build_lstm_model(input_shape, output_dim, learning_rate=0.001):
    """
    Build Bidirectional LSTM model for multi-output time series forecasting.
    
    Args:
        input_shape: Tuple (time_steps, features)
        output_dim: Number of output features
        learning_rate: Optimizer learning rate
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True, activation='tanh')),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(output_dim, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model


def build_gru_model(input_shape, output_dim, learning_rate=0.001):
    """
    Build Bidirectional GRU model for multi-output time series forecasting.
    
    Args:
        input_shape: Tuple (time_steps, features)
        output_dim: Number of output features
        learning_rate: Optimizer learning rate
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Bidirectional(GRU(128, return_sequences=True, activation='tanh'), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(GRU(64, return_sequences=True, activation='tanh')),
        Dropout(0.2),
        GRU(32, activation='tanh'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model


def build_cnn_model(input_shape, output_dim, learning_rate=0.001):
    """
    Build 1D CNN model for multi-output time series forecasting.
    
    Args:
        input_shape: Tuple (time_steps, features)
        output_dim: Number of output features
        learning_rate: Optimizer learning rate
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='causal', input_shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu', padding='causal'),
        Dropout(0.2),
        Conv1D(32, kernel_size=3, activation='relu', padding='causal'),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model


# ============================================================================
# SECTION 6: TRAINING UTILITIES
# ============================================================================

def get_callbacks():
    """Define training callbacks"""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train a Keras model"""
    callbacks = get_callbacks()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ============================================================================
# SECTION 7: EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred, set_name):
    """Calculate and display evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    print(f"\n=== {set_name} Set Metrics ===")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")
    print(f"MAPE: {mape:.4f}%")

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


# ============================================================================
# SECTION 8: PREDICTION AND FORECASTING
# ============================================================================

def make_predictions(models_dict, X_train, X_val, X_test):
    """Make predictions for all models"""
    predictions = {}
    
    for model_name, model in models_dict.items():
        predictions[model_name] = {
            'train': model.predict(X_train, verbose=0),
            'val': model.predict(X_val, verbose=0),
            'test': model.predict(X_test, verbose=0)
        }
    
    return predictions


def inverse_transform_predictions(predictions, scaler):
    """Inverse transform scaled predictions back to original scale"""
    return scaler.inverse_transform(predictions)


def forecast_future(model, last_sequence, n_steps, scaler):
    """
    Forecast future values using iterative prediction.
    
    Args:
        model: Trained model
        last_sequence: Last n_steps of data (scaled)
        n_steps: Number of future steps to predict
        scaler: Fitted scaler for inverse transformation
    
    Returns:
        Array of forecasted values (original scale)
    """
    forecasts = []
    current_sequence = last_sequence.copy()

    for _ in range(n_steps):
        input_seq = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        next_pred = model.predict(input_seq, verbose=0)[0]
        forecasts.append(next_pred)
        current_sequence = np.vstack([current_sequence[1:], next_pred])

    forecasts = np.array(forecasts)
    forecasts_original = scaler.inverse_transform(forecasts)
    return forecasts_original


# ============================================================================
# SECTION 9: VISUALIZATION UTILITIES
# ============================================================================

def plot_training_history(histories_dict):
    """Plot training history for multiple models"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    for name, hist in histories_dict.items():
        epochs = range(1, len(hist.history['loss']) + 1)
        axes[0].plot(epochs, hist.history['loss'], label=f'{name} Train')
        axes[0].plot(epochs, hist.history['val_loss'], label=f'{name} Val', linestyle='--')
    
    axes[0].set_title('Training Loss Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for name, hist in histories_dict.items():
        epochs = range(1, len(hist.history['mae']) + 1)
        axes[1].plot(epochs, hist.history['mae'], label=f'{name} Train')
        axes[1].plot(epochs, hist.history['val_mae'], label=f'{name} Val', linestyle='--')
    
    axes[1].set_title('Training MAE Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_overfitting(history, train_metrics, val_metrics, test_metrics):
    """Analyze overfitting through loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    epochs = range(1, len(history.history['loss']) + 1)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    axes[0].plot(epochs, train_loss, label='Training Loss', linewidth=2, color='blue')
    axes[0].plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='red')
    axes[0].fill_between(epochs, train_loss, val_loss, alpha=0.2, color='gray')
    axes[0].set_title('Overfitting Analysis: Loss Curves', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    metric_names = ['MSE', 'MAE', 'R²']
    train_vals = [train_metrics['MSE'], train_metrics['MAE'], train_metrics['R2']]
    val_vals = [val_metrics['MSE'], val_metrics['MAE'], val_metrics['R2']]
    test_vals = [test_metrics['MSE'], test_metrics['MAE'], test_metrics['R2']]

    x = np.arange(len(metric_names))
    width = 0.25

    axes[1].bar(x - width, train_vals, width, label='Training', color='blue', alpha=0.7)
    axes[1].bar(x, val_vals, width, label='Validation', color='red', alpha=0.7)
    axes[1].bar(x + width, test_vals, width, label='Test', color='green', alpha=0.7)

    axes[1].set_title('Bias-Variance Analysis', fontweight='bold')
    axes[1].set_xlabel('Metric')
    axes[1].set_ylabel('Value')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_names)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    final_gap = abs(train_loss[-1] - val_loss[-1])
    gap_percentage = (final_gap / train_loss[-1]) * 100

    print("\n=== Overfitting Metrics ===")
    print(f"Final Training Loss: {train_loss[-1]:.6f}")
    print(f"Final Validation Loss: {val_loss[-1]:.6f}")
    print(f"Generalization Gap: {final_gap:.6f} ({gap_percentage:.2f}%)")


# ============================================================================
# SECTION 10: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("="*70)
    print("DEEP LEARNING TIME-SERIES FORECASTING: CANADIAN EMPLOYMENT RATES")
    print("="*70)

    # Load data
    print("\n[1/10] Loading data...")
    df_original = load_data('example_data.csv')
    
    # Explore data
    print("\n[2/10] Exploring dataset...")
    explore_dataset(df_original)
    
    # Pivot data
    print("\n[3/10] Pivoting data...")
    df_pivot = pivot_employment_data(df_original)
    df_pivot.to_csv('df_pivot.csv', index=False)
    print(f"Pivoted dataset shape: {df_pivot.shape}")
    
    # Preprocess
    print("\n[4/10] Preprocessing data...")
    df_pivot['month'] = pd.to_datetime(df_pivot['month'])
    df_clean, numeric_cols = handle_missing_values(df_pivot)
    
    dates = df_clean['month'].values
    features = df_clean[numeric_cols].values
    
    scaled_features, scaler = scale_features(features)
    
    # Create sequences
    print("\n[5/10] Creating sequences...")
    N_STEPS = 12
    X, y = create_sequences(scaled_features, N_STEPS)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Build models
    print("\n[6/10] Building models...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_dim = y_train.shape[1]
    
    model_lstm = build_lstm_model(input_shape, output_dim)
    model_gru = build_gru_model(input_shape, output_dim)
    model_cnn = build_cnn_model(input_shape, output_dim)
    
    print("Models created successfully")
    
    # Train models
    print("\n[7/10] Training models...")
    print("Training LSTM...")
    history_lstm = train_model(model_lstm, X_train, y_train, X_val, y_val)
    
    print("\nTraining GRU...")
    history_gru = train_model(model_gru, X_train, y_train, X_val, y_val)
    
    print("\nTraining CNN...")
    history_cnn = train_model(model_cnn, X_train, y_train, X_val, y_val)
    
    # Evaluate models
    print("\n[8/10] Evaluating models...")
    models_dict = {'LSTM': model_lstm, 'GRU': model_gru, 'CNN': model_cnn}
    
    for model_name, model in models_dict.items():
        y_test_pred = model.predict(X_test, verbose=0)
        calculate_metrics(y_test, y_test_pred, f"Test ({model_name})")
    
    # Make forecasts
    print("\n[9/10] Generating forecasts...")
    last_known_sequence = scaled_features[-N_STEPS:]
    
    future_forecasts_lstm = forecast_future(model_lstm, last_known_sequence, 60, scaler)
    future_forecasts_gru = forecast_future(model_gru, last_known_sequence, 60, scaler)
    future_forecasts_cnn = forecast_future(model_cnn, last_known_sequence, 60, scaler)
    
    # Create forecast dataframes
    last_date = pd.to_datetime(dates[-1])
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=60, freq='MS')
    
    forecast_df_lstm = pd.DataFrame(future_forecasts_lstm, columns=numeric_cols)
    forecast_df_lstm.insert(0, 'month', forecast_dates)
    
    forecast_df_gru = pd.DataFrame(future_forecasts_gru, columns=numeric_cols)
    forecast_df_gru.insert(0, 'month', forecast_dates)
    
    forecast_df_cnn = pd.DataFrame(future_forecasts_cnn, columns=numeric_cols)
    forecast_df_cnn.insert(0, 'month', forecast_dates)
    
    # Save forecasts
    print("\n[10/10] Saving results...")
    forecast_df_lstm.to_csv('employment_forecasts_2020_2025_lstm.csv', index=False)
    forecast_df_gru.to_csv('employment_forecasts_2020_2025_gru.csv', index=False)
    forecast_df_cnn.to_csv('employment_forecasts_2020_2025_cnn.csv', index=False)
    
    # Save models
    model_lstm.save('employment_forecast_lstm_final.h5')
    model_gru.save('employment_forecast_gru_final.h5')
    model_cnn.save('employment_forecast_cnn_final.h5')
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n" + "="*70)
    print("PROJECT COMPLETE - ALL DELIVERABLES GENERATED")
    print("="*70)
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


if __name__ == "__main__":
    main()
