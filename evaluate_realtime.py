"""
Mede a acuracia real do modelo de acordes em condicoes de captura ao vivo
(janelas de RECORD_DURATION segundos, como em audio_capture.py).

Reusa o split de teste e o corte em janelas de train_model.py, garantindo que
a avaliacao usa exatamente a mesma logica usada no treino (sem duplicar codigo
que poderia divergir com o tempo).
"""
import time

import joblib
import numpy as np
import tensorflow as tf

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

assert abs(RECORD_DURATION - WINDOW_SECONDS) < 1e-6, (
    f"RECORD_DURATION (audio_capture.py, {RECORD_DURATION}s) != WINDOW_SECONDS "
    f"(train_model.py, {WINDOW_SECONDS}s) - ajuste os dois para o mesmo valor "
    "antes de avaliar, senao a medicao nao reflete a captura ao vivo real."
)


def predict_with_timing(segment, model, encoder, scaler):
    """Replica predict_note() mas mede separadamente extracao de features
    (Librosa) e inferencia do modelo."""
    t0 = time.perf_counter()
    features = extract_features(
        segment, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN, is_file=False
    )
    extraction_time = time.perf_counter() - t0

    if features is None:
        return None, extraction_time, None

    features_reshaped = features.reshape(-1, features.shape[-1])
    features_scaled = scaler.transform(features_reshaped).reshape(features.shape)
    features_final = np.swapaxes(features_scaled, 0, 1)
    features_final = np.expand_dims(features_final, axis=0)

    t1 = time.perf_counter()
    prediction = model.predict(features_final, verbose=0)
    inference_time = time.perf_counter() - t1

    predicted_idx = np.argmax(prediction[0])
    predicted_label = encoder.inverse_transform([predicted_idx])[0]
    return predicted_label, extraction_time, inference_time


def print_timing_block(title, times_seconds):
    times_ms = np.array(times_seconds) * 1000
    print(f"{title}")
    print(f"  media:   {times_ms.mean():.2f} ms")
    print(f"  mediana: {np.median(times_ms):.2f} ms")
    print(f"  p95:     {np.percentile(times_ms, 95):.2f} ms")


def main():
    print("Carregando modelo, encoder e scaler salvos...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    encoder = joblib.load(ENCODER_SAVE_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)

    print("Reconstruindo split de teste (20% nao usado no treino, mesmo random_state=42)...")
    _, test_files, _ = split_dataset_files()
    print(f"{len(test_files)} arquivos de teste identificados.\n")

    print(f"Simulando captura ao vivo: janelas de {RECORD_DURATION}s...\n")

    extraction_times = []
    inference_times = []
    correct = 0
    total = 0

    for file_path, true_label in test_files:
        segments = make_windowed_segments(file_path)
        for segment in segments:
            predicted, extraction_time, inference_time = predict_with_timing(segment, model, encoder, scaler)
            extraction_times.append(extraction_time)
            if predicted is None:
                continue
            inference_times.append(inference_time)
            total += 1
            if predicted == true_label:
                correct += 1

    print("=" * 64)
    print("RELATORIO - Acuracia e latencia em condicoes de captura ao vivo")
    print("=" * 64)
    print(f"Arquivos de teste utilizados:     {len(test_files)}")
    print(f"Segmentos de {RECORD_DURATION}s avaliados:       {total}")
    if total:
        print(f"Acuracia em segmentos curtos:      {correct / total:.4f}  ({correct}/{total})")
    else:
        print("Acuracia em segmentos curtos:      N/A (nenhum segmento valido)")
    print("-" * 64)
    print_timing_block("Tempo de extracao de features (Librosa/MFCC):", extraction_times)
    print("-" * 64)
    print_timing_block("Tempo de inferencia do modelo (model.predict):", inference_times)
    print("-" * 64)

    if extraction_times and inference_times:
        avg_total = np.mean(extraction_times) + np.mean(inference_times)
        print(f"Tempo total medio por predicao:   {avg_total * 1000:.2f} ms")
        print(f"Orcamento por janela (RECORD_DURATION): {RECORD_DURATION * 1000:.0f} ms")
        status = "OK (cabe dentro da janela)" if avg_total < RECORD_DURATION else "ESTOURA o orcamento da janela"
        print(f"Viabilidade para tempo real:       {status}")
    print("=" * 64)


if __name__ == "__main__":
    main()
