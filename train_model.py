import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import joblib
from pydub import AudioSegment

warnings.filterwarnings("ignore", category=FutureWarning)

DATASET_PATH = 'dataset'
MODEL_SAVE_PATH = 'trained_model/chord_recognizer_cnn_model.h5'
ENCODER_SAVE_PATH = 'trained_model/label_encoder_chords.joblib'
SCALER_SAVE_PATH = 'trained_model/scaler_chords.joblib'

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 704 


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

def load_dataset(dataset_path=DATASET_PATH):
    X = []
    y = []
    labels = sorted(os.listdir(dataset_path))
    valid_labels = []
    for label in labels:
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            valid_labels.append(label)
            print(f"Carregando áudios para o acorde: {label}")
            for filename in os.listdir(label_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(label_path, filename)
                    features = extract_features(file_path, is_file=True)
                    if features is not None:
                        X.append(features)
                        y.append(label)
    return np.array(X), np.array(y), sorted(valid_labels)

def train_model():
    print("Iniciando carregamento do dataset...")
    X, y, labels = load_dataset()

    if len(X) == 0:
        print("Nenhum dado encontrado no dataset. Verifique se a pasta 'dataset' contém subpastas com arquivos .wav de acordes.")
        return None, None, None

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    valid_indices = [i for i, features in enumerate(X) if features is not None]
    X_filtered = np.array([X[i] for i in valid_indices])
    y_filtered = np.array([y_categorical[i] for i in valid_indices])

    if len(X_filtered) == 0:
        print("Nenhum dado válido após extração de features. Verifique seus arquivos de áudio.")
        return None, None, None
        
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=np.argmax(y_filtered, axis=1))

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
    print(f"Acurácia final do modelo de acordes no conjunto de teste: {accuracy:.4f}")

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


def predict_note(audio_segment_np, model, encoder, scaler, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
  
    SILENCE_THRESHOLD = 0.003 
    if audio_segment_np.size > 0:
        segment_rms = np.sqrt(np.mean(audio_segment_np**2))
        if segment_rms < SILENCE_THRESHOLD:
            return "Silêncio"

    features = extract_features(audio_segment_np, sample_rate=sample_rate, n_mfcc=n_mfcc, max_pad_len=max_pad_len, is_file=False)
    
    if features is None:
        return "N/A - Áudio curto"

    features_reshaped_for_scaler = features.reshape(-1, features.shape[-1])
    features_scaled_reshaped = scaler.transform(features_reshaped_for_scaler)
    features_scaled = features_scaled_reshaped.reshape(features.shape)

    features_final = np.swapaxes(features_scaled, 0, 1)
    features_final = np.expand_dims(features_final, axis=0)

    prediction = model.predict(features_final, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    
    if predicted_class_idx < len(encoder.classes_):
        predicted_chord = encoder.inverse_transform([predicted_class_idx])[0]
    else:
        predicted_chord = "Acorde Desconhecido"

    return predicted_chord

if __name__ == '__main__':
    train_model()
