import os

import joblib
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from train_model import build_windowed_dataset, split_dataset_files

OUT_DIR = 'trained_model_experiment'


def main():
    print("Listando e dividindo arquivos do dataset...")
    train_files, test_files, labels = split_dataset_files()
    labels = sorted(labels)

    X_train, y_train_raw = build_windowed_dataset(train_files)
    X_test, y_test_raw = build_windowed_dataset(test_files)
    print(f"Treino: {len(X_train)}  Teste: {len(X_test)}")

    encoder = LabelEncoder()
    encoder.fit(labels)
    y_train = to_categorical(encoder.transform(y_train_raw), num_classes=len(labels))
    y_test = to_categorical(encoder.transform(y_test_raw), num_classes=len(labels))

    original_shape = X_train.shape
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(original_shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

    X_train_final = np.swapaxes(X_train_scaled, 1, 2)
    X_test_final = np.swapaxes(X_test_scaled, 1, 2)

    bdim_idx = int(encoder.transform(['Bdim'])[0])
    class_weight = {i: 1.0 for i in range(len(labels))}
    class_weight[bdim_idx] = 2.5
    print(f"Pesos de classe: {dict(zip(encoder.classes_, [class_weight[i] for i in range(len(labels))]))}")

    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu',
               input_shape=(X_train_final.shape[1], X_train_final.shape[2])),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(len(labels), activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

    print("Treinando com peso extra para Bdim...")
    model.fit(
        X_train_final, y_train,
        epochs=200, batch_size=32,
        validation_data=(X_test_final, y_test),
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight,
        verbose=2,
    )

    loss, accuracy = model.evaluate(X_test_final, y_test, verbose=0)
    print(f"\nAcuracia geral (val split padrao): {accuracy:.4f}")

    y_pred_idx = np.argmax(model.predict(X_test_final, verbose=0), axis=1)
    y_true_idx = np.argmax(y_test, axis=1)
    y_pred_labels = encoder.inverse_transform(y_pred_idx)
    y_true_labels = encoder.inverse_transform(y_true_idx)

    print("\n" + "=" * 64)
    print("MATRIZ DE CONFUSAO - MODELO EXPERIMENTAL (peso extra Bdim)")
    print("=" * 64)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
    header = "        " + "".join(f"{l:>6}" for l in labels)
    print(header)
    for label, row in zip(labels, cm):
        print(f"{label:>7} " + "".join(f"{v:>6}" for v in row))

    print()
    print(classification_report(y_true_labels, y_pred_labels, labels=labels, zero_division=0))

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(f'{OUT_DIR}/chord_recognizer_cnn_model.h5')
    joblib.dump(encoder, f'{OUT_DIR}/label_encoder_chords.joblib')
    joblib.dump(scaler, f'{OUT_DIR}/scaler_chords.joblib')
    print(f"\nModelo experimental salvo em {OUT_DIR}/ (nao substitui o modelo em producao)")


if __name__ == '__main__':
    main()
