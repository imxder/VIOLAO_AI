import os

import joblib
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from train_model import (
    ENCODER_SAVE_PATH,
    MODEL_SAVE_PATH,
    SCALER_SAVE_PATH,
    build_windowed_dataset,
    split_dataset_files,
)

OUT_DIR = 'trained_model_ensemble'
N_EXTRA_MODELS = 2


def build_model(input_shape, n_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    print("Listando e dividindo arquivos do dataset (mesmo split do modelo em producao)...")
    train_files, test_files, labels = split_dataset_files()
    labels = sorted(labels)

    X_train, y_train_raw = build_windowed_dataset(train_files)
    X_test, y_test_raw = build_windowed_dataset(test_files)

    encoder = LabelEncoder()
    encoder.fit(labels)
    y_train = to_categorical(encoder.transform(y_train_raw), num_classes=len(labels))
    y_test = to_categorical(encoder.transform(y_test_raw), num_classes=len(labels))

    original_shape = X_train.shape
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(original_shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    X_train_final = np.swapaxes(X_train_scaled, 1, 2)
    X_test_final = np.swapaxes(X_test_scaled, 1, 2)

    print("\nCarregando o modelo de producao atual como membro 1 do ensemble...")
    prod_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    prod_encoder = joblib.load(ENCODER_SAVE_PATH)
    prod_scaler = joblib.load(SCALER_SAVE_PATH)
    X_test_prod_scaled = prod_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    X_test_prod_final = np.swapaxes(X_test_prod_scaled, 1, 2)
    prod_probs = prod_model.predict(X_test_prod_final, verbose=0)
    # Realinha as colunas do modelo de producao para a ordem alfabetica usada aqui.
    prod_order_fix = [list(prod_encoder.classes_).index(l) for l in labels]
    prod_probs = prod_probs[:, prod_order_fix]

    all_probs = [prod_probs]

    for i in range(N_EXTRA_MODELS):
        print(f"\nTreinando modelo extra {i + 1}/{N_EXTRA_MODELS} (seed diferente)...")
        tf.random.set_seed(100 + i)
        model = build_model((X_train_final.shape[1], X_train_final.shape[2]), len(labels))
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
        model.fit(
            X_train_final, y_train,
            epochs=200, batch_size=32,
            validation_data=(X_test_final, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0,
        )
        loss, acc = model.evaluate(X_test_final, y_test, verbose=0)
        print(f"Modelo extra {i + 1} isolado: acuracia={acc:.4f}")
        all_probs.append(model.predict(X_test_final, verbose=0))

    print("\nCombinando previsoes (media simples das probabilidades)...")
    ensemble_probs = np.mean(all_probs, axis=0)
    y_pred_idx = np.argmax(ensemble_probs, axis=1)
    y_true_idx = np.argmax(y_test, axis=1)
    y_pred_labels = encoder.inverse_transform(y_pred_idx)
    y_true_labels = encoder.inverse_transform(y_true_idx)

    ensemble_acc = np.mean(y_pred_idx == y_true_idx)

    print("\n" + "=" * 64)
    print(f"ACURACIA DO ENSEMBLE ({1 + N_EXTRA_MODELS} modelos): {ensemble_acc:.4f}")
    print("=" * 64)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
    header = "        " + "".join(f"{l:>6}" for l in labels)
    print(header)
    for label, row in zip(labels, cm):
        print(f"{label:>7} " + "".join(f"{v:>6}" for v in row))
    print()
    print(classification_report(y_true_labels, y_pred_labels, labels=labels, zero_division=0))


if __name__ == '__main__':
    main()
