"""Training helpers and callbacks."""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def get_callbacks():
    """Define training callbacks."""
    return [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001, verbose=1),
        ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, verbose=1),
    ]


def train_model(model, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 32):
    """Train a Keras model."""
    callbacks = get_callbacks()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history
