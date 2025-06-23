import sounddevice as sd
import numpy as np
import threading
import time
import queue

SAMPLE_RATE = 22050  
BLOCK_SIZE = 1024    
CHANNELS = 1         
DTYPE = 'float32'    
RECORD_DURATION = 0.75 

BUFFER_SIZE_SECONDS = 3 
audio_buffer = np.zeros(int(SAMPLE_RATE * BUFFER_SIZE_SECONDS), dtype=DTYPE)
buffer_write_idx = 0
buffer_lock = threading.Lock() 

stream = None
is_recording = False

def callback(indata, frames, time, status):
    global audio_buffer, buffer_write_idx, buffer_lock

    if status:
        print(f"Status do stream de áudio: {status}") 
    
    with buffer_lock:
        num_samples = indata.shape[0]
        if buffer_write_idx + num_samples > audio_buffer.size:
            samples_to_end = audio_buffer.size - buffer_write_idx
            audio_buffer[buffer_write_idx:] = indata[:samples_to_end, 0]
            audio_buffer[:num_samples - samples_to_end] = indata[samples_to_end:, 0]
            buffer_write_idx = num_samples - samples_to_end
        else:
            audio_buffer[buffer_write_idx : buffer_write_idx + num_samples] = indata[:, 0]
            buffer_write_idx += num_samples

def list_audio_devices():
    try:
        devices = sd.query_devices()
        return devices
    except Exception as e:
        print(f"Erro ao listar dispositivos de áudio: {e}")
        return []

def start_recording(device_id=None):
    global stream, is_recording, audio_buffer, buffer_write_idx

    if is_recording:
        return True

    print(f"Iniciando gravação com dispositivo ID: {device_id if device_id is not None else 'Padrão'}")
    try:
        audio_buffer = np.zeros(int(SAMPLE_RATE * BUFFER_SIZE_SECONDS), dtype=DTYPE)
        buffer_write_idx = 0

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=device_id,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=callback
        )
        stream.start()
        is_recording = True
        print("Gravação iniciada.")
        return True
    except Exception as e:
        print(f"Erro ao iniciar gravação: {e}")
        is_recording = False
        return False

def stop_recording():
    global stream, is_recording
    if stream is not None and stream.active:
        stream.stop()
        stream.close()
        is_recording = False
        print("Gravação parada.")

def get_audio_segment():
    global audio_buffer, buffer_write_idx, buffer_lock

    num_samples_needed = int(SAMPLE_RATE * RECORD_DURATION)

    with buffer_lock:
        if audio_buffer.size < num_samples_needed:
            return np.array([])

        if buffer_write_idx >= num_samples_needed:
            segment = audio_buffer[buffer_write_idx - num_samples_needed : buffer_write_idx].copy()
        else:
            part1_len = num_samples_needed - buffer_write_idx
            part1 = audio_buffer[audio_buffer.size - part1_len:].copy()
            part2 = audio_buffer[0:buffer_write_idx].copy()
            segment = np.concatenate((part1, part2))
    
    return segment

if __name__ == '__main__':
    print("Executando audio_capture.py diretamente. Isso geralmente lista os dispositivos de áudio.")
    devices = list_audio_devices()
    if devices:
        print("\nDispositivos de Áudio Disponíveis:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  ID: {device['index']}, Nome: {device['name']}, Canais de Entrada: {device['max_input_channels']}")
        print("\nPara usar um dispositivo específico, passe o ID para a função start_recording().")
    else:
        print("Nenhum dispositivo de áudio encontrado ou erro ao listar.")