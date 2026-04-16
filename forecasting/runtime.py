"""Runtime and environment setup helpers."""

import os
import warnings


def configure_runtime() -> None:
    """Configure TensorFlow runtime flags before importing tensorflow."""
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    warnings.filterwarnings("ignore")


def print_runtime_info(tf_module, keras_module) -> None:
    """Print runtime diagnostics for TensorFlow and Keras."""
    keras_version = getattr(keras_module, "__version__", getattr(tf_module.keras, "__version__", "legacy-tf-keras"))
    print(f"TensorFlow Version: {tf_module.__version__}")
    print(f"Keras Version: {keras_version}")
    print(f"GPU Available: {tf_module.config.list_physical_devices('GPU')}")
