import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from audio_capture import RECORD_DURATION
from train_model import (
    ENCODER_SAVE_PATH,
    MAX_PAD_LEN,
    MODEL_SAVE_PATH,
    N_MFCC,
    SAMPLE_RATE,
    SCALER_SAVE_PATH,
    WINDOW_SECONDS,
    extract_features,
    make_windowed_segments,
    split_dataset_files,
)

assert abs(RECORD_DURATION - WINDOW_SECONDS) < 1e-6


def predict(segment, model, encoder, scaler):
    features = extract_features(
        segment, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN, is_file=False
    )
    if features is None:
        return None
    features_reshaped = features.reshape(-1, features.shape[-1])
    features_scaled = scaler.transform(features_reshaped).reshape(features.shape)
    features_final = np.swapaxes(features_scaled, 0, 1)
    features_final = np.expand_dims(features_final, axis=0)
    prediction = model.predict(features_final, verbose=0)
    predicted_idx = np.argmax(prediction[0])
    return encoder.inverse_transform([predicted_idx])[0]


def main():
    print("Carregando modelo, encoder e scaler salvos...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    encoder = joblib.load(ENCODER_SAVE_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)

    _, test_files, labels = split_dataset_files()
    labels = sorted(labels)

    y_true = []
    y_pred = []
    for file_path, true_label in test_files:
        for segment in make_windowed_segments(file_path):
            predicted = predict(segment, model, encoder, scaler)
            if predicted is None:
                continue
            y_true.append(true_label)
            y_pred.append(predicted)

    print()
    print("=" * 64)
    print("MATRIZ DE CONFUSAO (linhas = esperado, colunas = previsto)")
    print("=" * 64)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = "        " + "".join(f"{l:>6}" for l in labels)
    print(header)
    for label, row in zip(labels, cm):
        print(f"{label:>7} " + "".join(f"{v:>6}" for v in row))

    print()
    print("=" * 64)
    print("RELATORIO POR CLASSE")
    print("=" * 64)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()
