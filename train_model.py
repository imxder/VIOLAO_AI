import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

DATASET_PATH = 'dataset'
MODEL_SAVE_PATH = 'trained_model/chord_recognizer_cnn_model.h5'
ENCODER_SAVE_PATH = 'trained_model/label_encoder_chords.joblib'
SCALER_SAVE_PATH = 'trained_model/scaler_chords.joblib'

SAMPLE_RATE = 22050
N_MFCC = 40

# WINDOW_SECONDS precisa ser igual a RECORD_DURATION em audio_capture.py: o
# modelo so acerta ao vivo se treinar com o mesmo tamanho de janela que vai
# receber na captura em tempo real.
WINDOW_SECONDS = 1.5
CROP_HOP_SECONDS = WINDOW_SECONDS / 2  # 50% de sobreposicao entre janelas de treino
MAX_CROPS_PER_FILE = 8  # evita que arquivos longos dominem o dataset
MIN_VALID_DURATION = 0.05


def _compute_max_pad_len(window_seconds=WINDOW_SECONDS, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC):
    dummy = np.zeros(int(sample_rate * window_seconds), dtype=np.float32)
    return librosa.feature.mfcc(y=dummy, sr=sample_rate, n_mfcc=n_mfcc).shape[1]


MAX_PAD_LEN = _compute_max_pad_len()


def extract_features(audio_data, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN, is_file=True):
    try:
        if is_file:
            audio, sr = librosa.load(audio_data, sr=sample_rate)
        else:
            audio = audio_data
            sr = sample_rate

        if len(audio) < sample_rate * 0.05:
            return None

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        return mfccs
    except Exception as e:
        print(f"Erro ao processar áudio para features: {e}")
        return None


def list_dataset_files(dataset_path=DATASET_PATH):
    """Lista (file_path, label) de todo o dataset, com a mesma filtragem por
    duracao minima usada no treino (sem decodificar o audio inteiro)."""
    file_paths = []
    labels = []
    valid_labels = sorted(
        d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))
    )
    for label in valid_labels:
        label_path = os.path.join(dataset_path, label)
        for filename in os.listdir(label_path):
            if not filename.endswith(".wav"):
                continue
            file_path = os.path.join(label_path, filename)
            try:
                info = sf.info(file_path)
                duration = info.frames / info.samplerate
            except Exception:
                continue
            if duration >= MIN_VALID_DURATION:
                file_paths.append(file_path)
                labels.append(label)
    return file_paths, labels, valid_labels


def split_dataset_files(dataset_path=DATASET_PATH, test_size=0.2, random_state=42):
    """Divide por ARQUIVO (nao por janela/crop), para nenhum recorte do mesmo
    audio vazar entre treino e teste."""
    file_paths, labels, valid_labels = list_dataset_files(dataset_path)
    train_idx, test_idx = train_test_split(
        np.arange(len(file_paths)), test_size=test_size, random_state=random_state, stratify=labels
    )
    train_files = [(file_paths[i], labels[i]) for i in train_idx]
    test_files = [(file_paths[i], labels[i]) for i in test_idx]
    return train_files, test_files, valid_labels


def make_windowed_segments(file_path, window_seconds=WINDOW_SECONDS, hop_seconds=CROP_HOP_SECONDS,
                            sample_rate=SAMPLE_RATE, max_crops=MAX_CROPS_PER_FILE):
    """Recorta o arquivo em janelas deslizantes do tamanho da captura ao vivo,
    simulando os offsets que get_audio_segment() entrega em audio_capture.py."""
    audio, _ = librosa.load(file_path, sr=sample_rate)
    window_len = int(sample_rate * window_seconds)

    if len(audio) < window_len:
        audio = np.pad(audio, (0, window_len - len(audio)))

    hop_len = max(1, int(sample_rate * hop_seconds))
    starts = list(range(0, len(audio) - window_len + 1, hop_len))
    if not starts:
        starts = [0]

    if len(starts) > max_crops:
        idx = sorted(set(np.linspace(0, len(starts) - 1, max_crops).round().astype(int)))
        starts = [starts[i] for i in idx]

    return [audio[s:s + window_len] for s in starts]


def build_windowed_dataset(file_label_pairs):
    X, y = [], []
    for file_path, label in file_label_pairs:
        for segment in make_windowed_segments(file_path):
            features = extract_features(segment, is_file=False)
            if features is not None:
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)


def train_model():
    print("Listando e dividindo arquivos do dataset (split por arquivo, 80/20)...")
    train_files, test_files, labels = split_dataset_files()

    if not train_files:
        print("Nenhum dado encontrado no dataset. Verifique se a pasta 'dataset' contém subpastas com arquivos .wav de acordes.")
        return None, None, None

    print(f"Gerando janelas de {WINDOW_SECONDS}s (= RECORD_DURATION em audio_capture.py, hop={CROP_HOP_SECONDS}s, max {MAX_CROPS_PER_FILE} crops/arquivo)...")
    X_train, y_train_raw = build_windowed_dataset(train_files)
    X_test, y_test_raw = build_windowed_dataset(test_files)
    print(f"Exemplos de treino: {len(X_train)} (de {len(train_files)} arquivos)")
    print(f"Exemplos de teste:  {len(X_test)} (de {len(test_files)} arquivos)")

    if len(X_train) == 0 or len(X_test) == 0:
        print("Nenhum dado válido após extração de features. Verifique seus arquivos de áudio.")
        return None, None, None

    encoder = LabelEncoder()
    encoder.fit(labels)
    y_train = to_categorical(encoder.transform(y_train_raw), num_classes=len(labels))
    y_test = to_categorical(encoder.transform(y_test_raw), num_classes=len(labels))

    original_shape = X_train.shape
    X_train_reshaped_for_scaler = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped_for_scaler = X_test.reshape(-1, X_test.shape[-1])

    scaler = StandardScaler()
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped_for_scaler)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped_for_scaler)

    X_train_scaled = X_train_scaled_reshaped.reshape(original_shape)
    X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)

    X_train_final = np.swapaxes(X_train_scaled, 1, 2)
    X_test_final = np.swapaxes(X_test_scaled, 1, 2)


    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(X_train_final.shape[1], X_train_final.shape[2])),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(len(labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Shape final de X_train para o modelo: {X_train_final.shape}")
    print(f"Shape de y_train: {y_train.shape}")
    print(model.summary())

    print("Iniciando treinamento do modelo de acordes (CNN)...")

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

    history = model.fit(X_train_final, y_train,
                        epochs=200,
                        batch_size=32,
                        validation_data=(X_test_final, y_test),
                        callbacks=[early_stopping, reduce_lr],
                        verbose=1)

    loss, accuracy = model.evaluate(X_test_final, y_test, verbose=0)
    print(f"Acurácia final do modelo de acordes no conjunto de teste (janelas de {WINDOW_SECONDS}s): {accuracy:.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    joblib.dump(encoder, ENCODER_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Modelo de acordes (CNN), encoder e scaler salvos em: {os.path.dirname(MODEL_SAVE_PATH)}")

    return model, encoder, scaler

def load_trained_model():
    if not os.path.exists(MODEL_SAVE_PATH) or \
       not os.path.exists(ENCODER_SAVE_PATH) or \
       not os.path.exists(SCALER_SAVE_PATH):
        print(f"Arquivos do modelo/encoder/scaler para acordes não encontrados em 'trained_model/'.")
        print("Treinando um novo modelo de acordes...")
        return train_model()
    else:
        print(f"Carregando modelo de acordes de: {MODEL_SAVE_PATH}")
        try:
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            encoder = joblib.load(ENCODER_SAVE_PATH)
            scaler = joblib.load(SCALER_SAVE_PATH)
            print("Modelo de acordes (CNN), encoder e scaler carregados com sucesso!")
            return model, encoder, scaler
        except Exception as e:
            print(f"Erro ao carregar o modelo de acordes: {e}")
            print("Pode ser necessário retreinar o modelo ou verificar a integridade dos arquivos.")
            return None, None, None


SILENCE_THRESHOLD = 0.003


def _predict_raw(audio_segment_np, model, encoder, scaler, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    """Nucleo compartilhado de inferencia. Retorna (chord_or_None, confidence, status),
    onde status in {'silence', 'too_short', 'ok'}."""
    if audio_segment_np.size > 0:
        segment_rms = np.sqrt(np.mean(audio_segment_np ** 2))
        if segment_rms < SILENCE_THRESHOLD:
            return None, 0.0, 'silence'

    features = extract_features(audio_segment_np, sample_rate=sample_rate, n_mfcc=n_mfcc, max_pad_len=max_pad_len, is_file=False)
    if features is None:
        return None, 0.0, 'too_short'

    features_reshaped_for_scaler = features.reshape(-1, features.shape[-1])
    features_scaled_reshaped = scaler.transform(features_reshaped_for_scaler)
    features_scaled = features_scaled_reshaped.reshape(features.shape)

    features_final = np.swapaxes(features_scaled, 0, 1)
    features_final = np.expand_dims(features_final, axis=0)

    prediction = model.predict(features_final, verbose=0)[0]
    predicted_class_idx = int(np.argmax(prediction))
    confidence = float(prediction[predicted_class_idx])

    if predicted_class_idx < len(encoder.classes_):
        predicted_chord = encoder.inverse_transform([predicted_class_idx])[0]
    else:
        predicted_chord = None

    return predicted_chord, confidence, 'ok'


def predict_note(audio_segment_np, model, encoder, scaler, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    chord, _, status = _predict_raw(audio_segment_np, model, encoder, scaler, sample_rate, n_mfcc, max_pad_len)
    if status == 'silence':
        return "Silêncio"
    if status == 'too_short':
        return "N/A - Áudio curto"
    return chord if chord is not None else "Acorde Desconhecido"


def recognize_chord(audio_segment_np, model, encoder, scaler, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    """Usado pela API HTTP: retorna um dict serializavel em JSON para o cliente Flutter."""
    chord, confidence, status = _predict_raw(audio_segment_np, model, encoder, scaler, sample_rate, n_mfcc, max_pad_len)
    if status in ('silence', 'too_short'):
        return {'status': status, 'chord': None, 'confidence': 0.0}
    if chord is None:
        return {'status': 'unknown', 'chord': None, 'confidence': round(confidence, 4)}
    return {'status': 'ok', 'chord': str(chord), 'confidence': round(confidence, 4)}

if __name__ == '__main__':
    train_model()
