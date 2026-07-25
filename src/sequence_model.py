import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

WINDOW_LEN = 6


def build_sequences(df, feat_df, window_len=WINDOW_LEN):
    """Returns (windows, last_row_index_per_window) sorted by entity + time."""
    df = df.reset_index(drop=True)
    windows, last_idx = [], []
    for entity_id, g in df.groupby("entity_id"):
        order = g.sort_values("timestamp").index.to_list()
        feats = feat_df.loc[order].values
        for i in range(len(order)):
            start = max(0, i - window_len + 1)
            window = feats[start:i + 1]
            if len(window) < window_len:
                pad = np.zeros((window_len - len(window), feats.shape[1]))
                window = np.vstack([pad, window])
            windows.append(window)
            last_idx.append(order[i])
    return np.array(windows, dtype="float32"), np.array(last_idx)


def train_autoencoder(windows, benign_mask, epochs=15, batch_size=64):
    n_features = windows.shape[2]
    train_windows = windows[benign_mask]

    inputs = layers.Input(shape=(WINDOW_LEN, n_features))
    x = layers.LSTM(16, activation="tanh", return_sequences=False)(inputs)
    x = layers.RepeatVector(WINDOW_LEN)(x)
    x = layers.LSTM(16, activation="tanh", return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    autoencoder = models.Model(inputs, outputs)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        train_windows, train_windows,
        epochs=epochs, batch_size=batch_size, verbose=0,
        validation_split=0.1,
    )
    return autoencoder


def sequence_anomaly_scores(autoencoder, windows):
    recon = autoencoder.predict(windows, verbose=0)
    mse = np.mean(np.square(windows - recon), axis=(1, 2))
    score = (mse - mse.min()) / (mse.max() - mse.min() + 1e-9)
    return score


def compute_sequence_scores(df, feat_df, train_mask):
    """train_mask restricts which windows the autoencoder is allowed to TRAIN on
    (benign windows from the train time period only). It still SCORES windows
    from the eval period — that's the whole point of the held-out evaluation."""
    windows, last_idx = build_sequences(df, feat_df)
    is_benign = (df.loc[last_idx, "signal_type"] == "Benign").values
    is_train = train_mask.loc[last_idx].values
    fit_mask = is_benign & is_train

    autoencoder = train_autoencoder(windows, fit_mask)
    scores = sequence_anomaly_scores(autoencoder, windows)

    result = pd.Series(0.0, index=df.index)
    result.loc[last_idx] = scores
    return result, autoencoder
