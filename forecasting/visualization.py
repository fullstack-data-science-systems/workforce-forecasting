"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_training_history(histories_dict):
    """Plot training history for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for name, hist in histories_dict.items():
        epochs = range(1, len(hist.history["loss"]) + 1)
        axes[0].plot(epochs, hist.history["loss"], label=f"{name} Train")
        axes[0].plot(epochs, hist.history["val_loss"], label=f"{name} Val", linestyle="--")

    axes[0].set_title("Training Loss Comparison")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, hist in histories_dict.items():
        epochs = range(1, len(hist.history["mae"]) + 1)
        axes[1].plot(epochs, hist.history["mae"], label=f"{name} Train")
        axes[1].plot(epochs, hist.history["val_mae"], label=f"{name} Val", linestyle="--")

    axes[1].set_title("Training MAE Comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_overfitting(history, train_metrics, val_metrics, test_metrics):
    """Analyze overfitting through loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    epochs = range(1, len(history.history["loss"]) + 1)
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    axes[0].plot(epochs, train_loss, label="Training Loss", linewidth=2, color="blue")
    axes[0].plot(epochs, val_loss, label="Validation Loss", linewidth=2, color="red")
    axes[0].fill_between(epochs, train_loss, val_loss, alpha=0.2, color="gray")
    axes[0].set_title("Overfitting Analysis: Loss Curves", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    metric_names = ["MSE", "MAE", "R²"]
    train_vals = [train_metrics["MSE"], train_metrics["MAE"], train_metrics["R2"]]
    val_vals = [val_metrics["MSE"], val_metrics["MAE"], val_metrics["R2"]]
    test_vals = [test_metrics["MSE"], test_metrics["MAE"], test_metrics["R2"]]

    x = np.arange(len(metric_names))
    width = 0.25

    axes[1].bar(x - width, train_vals, width, label="Training", color="blue", alpha=0.7)
    axes[1].bar(x, val_vals, width, label="Validation", color="red", alpha=0.7)
    axes[1].bar(x + width, test_vals, width, label="Test", color="green", alpha=0.7)

    axes[1].set_title("Bias-Variance Analysis", fontweight="bold")
    axes[1].set_xlabel("Metric")
    axes[1].set_ylabel("Value")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_names)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    final_gap = abs(train_loss[-1] - val_loss[-1])
    gap_percentage = (final_gap / train_loss[-1]) * 100

    print("\n=== Overfitting Metrics ===")
    print(f"Final Training Loss: {train_loss[-1]:.6f}")
    print(f"Final Validation Loss: {val_loss[-1]:.6f}")
    print(f"Generalization Gap: {final_gap:.6f} ({gap_percentage:.2f}%)")
