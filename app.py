from flask import Flask, render_template, request, jsonify
import threading
import time
import os
import numpy as np
from collections import deque, Counter # Importa 'deque' e 'Counter'

from train_model import load_trained_model, predict_note, extract_features, SAMPLE_RATE, N_MFCC, MAX_PAD_LEN
from audio_capture import start_recording, stop_recording, get_audio_segment, list_audio_devices, SAMPLE_RATE as AUDIO_SAMPLE_RATE

app = Flask(__name__)

model_loaded = None
encoder_loaded = None
scaler_loaded = None
prediction_thread = None
prediction_active = False
current_note = "Aguardando áudio..."

PREDICTION_BUFFER_SIZE = 8   
STABILITY_THRESHOLD = 8  
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
stable_note = "Aguardando áudio..."


with app.app_context():
    print("Tentando carregar modelo de ACORDES na inicialização do Flask...")
    try:
        model_loaded, encoder_loaded, scaler_loaded = load_trained_model()
        if model_loaded is None:
            print("AVISO: Modelo de acordes não foi carregado. Verifique o dataset e o treinamento.")
            print("Para treinar o modelo, execute: python model.py")
        else:
            print("Modelo de acordes, encoder e scaler carregados com sucesso!")
    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar o modelo de acordes na inicialização: {e}")
        print("Certifique-se de que o modelo de acordes foi treinado e os arquivos .joblib estão na pasta 'trained_model'.")
        model_loaded = None 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list_audio_devices')
def get_audio_devices():
    devices = list_audio_devices()
    device_list = [{'id': d['index'], 'name': d['name'], 'max_input_channels': d['max_input_channels']} for d in devices]
    return jsonify(device_list)

def audio_prediction_loop():
    global current_note, prediction_active, prediction_buffer, stable_note

    if model_loaded is None or encoder_loaded is None or scaler_loaded is None:
        current_note = "Modelo não carregado. Treine o modelo primeiro."
        prediction_active = False
        return

    print("Iniciando loop de predição de áudio com filtro de estabilidade...")
    while prediction_active:
        try:
            audio_segment = get_audio_segment()

            if audio_segment.size > 0:

                predicted_chord = predict_note(audio_segment, model_loaded, encoder_loaded, scaler_loaded,
                                              sample_rate=AUDIO_SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN)
                
                if predicted_chord and "N/A" not in predicted_chord:
                    prediction_buffer.append(predicted_chord)

                if len(prediction_buffer) >= STABILITY_THRESHOLD:
                   
                    counts = Counter(prediction_buffer)
                    most_common_chord, num_occurrences = counts.most_common(1)[0]

                    if num_occurrences >= STABILITY_THRESHOLD:
                        if most_common_chord == "Silêncio" and stable_note != "Silêncio...":
                             if num_occurrences >= PREDICTION_BUFFER_SIZE -1:
                                stable_note = "Silêncio..."
                        elif most_common_chord != "Silêncio":
                             stable_note = most_common_chord
            
            current_note = stable_note

        except Exception as e:
            current_note = f"Erro na predição: {e}"
            print(f"Erro no loop de predição: {e}")
            prediction_active = False 
        
        time.sleep(0.08) 

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    
    global prediction_thread, prediction_active, current_note, prediction_buffer, stable_note

    if model_loaded is None or encoder_loaded is None or scaler_loaded is None:
        return jsonify(status='error', message='Modelo de reconhecimento não carregado. Treine o modelo primeiro.')

    if prediction_active:
        return jsonify(status='warning', message='Reconhecimento já está ativo.')

    data = request.get_json()
    device_id_str = data.get('device_id')
    try:
        device_id = int(device_id_str) 
    except (TypeError, ValueError):
        device_id = None 

    import audio_capture as ac
    if not ac.start_recording(device_id=device_id):
        return jsonify(status='error', message='Não foi possível iniciar a gravação. Verifique o dispositivo.')

    # Reseta o estado para um início limpo
    prediction_buffer.clear()
    stable_note = "Ouvindo..."
    current_note = "Ouvindo..."
    prediction_active = True
    
    prediction_thread = threading.Thread(target=audio_prediction_loop)
    prediction_thread.daemon = True
    prediction_thread.start()
    return jsonify(status='success', message='Reconhecimento iniciado.')

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    
    global prediction_active, current_note, stable_note, prediction_buffer

    prediction_active = False
    import audio_capture as ac
    ac.stop_recording()
    if prediction_thread and prediction_thread.is_alive():
        prediction_thread.join(timeout=1)
    
    prediction_buffer.clear()
    stable_note = "Reconhecimento parado."
    current_note = "Reconhecimento parado."
    
    return jsonify(status='success', message='Reconhecimento parado.')

@app.route('/get_note')
def get_note():
    global current_note
    
    if not prediction_active and current_note == "Reconhecimento parado.":
        return jsonify(status='stopped')
    return jsonify(note=current_note, status='active' if prediction_active else 'inactive')

if __name__ == '__main__':
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('dataset', exist_ok=True) 
    
    app.run(debug=True, threaded=True, host='0.0.0.0')