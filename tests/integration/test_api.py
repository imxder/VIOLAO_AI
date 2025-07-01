import pytest
from unittest.mock import patch

# Mock para os dados retornados pelos dispositivos de áudio
MOCK_AUDIO_DEVICES = [
    {'index': 0, 'name': 'Microfone Padrão', 'max_input_channels': 1},
    {'index': 1, 'name': 'Interface de Áudio', 'max_input_channels': 2},
    {'index': 2, 'name': 'Saída Estéreo (Loopback)', 'max_input_channels': 0}, # Dispositivo de saída
]

def test_list_audio_devices_endpoint(client, mocker):
    """
    Testa se o endpoint /list_audio_devices retorna uma lista JSON
    de dispositivos de entrada de áudio.
    """
    # Simula a função 'list_audio_devices' para retornar nossa lista mock
    mocker.patch('app.list_audio_devices', return_value=MOCK_AUDIO_DEVICES)

    response = client.get('/list_audio_devices')
    json_data = response.get_json()

    assert response.status_code == 200
    assert isinstance(json_data, list)
    # A rota da API retorna todos os dispositivos; a filtragem é feita no front-end.
    # Portanto, esperamos receber todos os 3 dispositivos da nossa lista mock.
    assert len(json_data) == 3
    assert json_data[0]['name'] == 'Microfone Padrão'
    assert json_data[1]['id'] == 1

def test_start_and_stop_recognition_flow(client, mocker):
    """
    Testa o fluxo completo de iniciar e parar o reconhecimento,
    verificando as mudanças de estado.
    """
    # Mock das funções que interagem com hardware e modelos
    mock_start_recording = mocker.patch('audio_capture.start_recording', return_value=True)
    mock_stop_recording = mocker.patch('audio_capture.stop_recording')
    # Garante que o modelo seja considerado "carregado" para o teste
    mocker.patch('app.model_loaded', True)
    mocker.patch('app.encoder_loaded', True)
    mocker.patch('app.scaler_loaded', True)

    # 1. Iniciar o reconhecimento
    start_response = client.post('/start_recognition', json={'device_id': '0'})
    start_data = start_response.get_json()

    assert start_response.status_code == 200
    assert start_data['status'] == 'success'
    mock_start_recording.assert_called_once_with(device_id=0)

    # 2. Verificar o estado (deve estar ativo)
    note_response_active = client.get('/get_note')
    note_data_active = note_response_active.get_json()
    assert note_data_active['status'] == 'active'
    assert note_data_active['note'] == 'Ouvindo...'

    # 3. Parar o reconhecimento
    stop_response = client.post('/stop_recognition')
    stop_data = stop_response.get_json()

    assert stop_response.status_code == 200
    assert stop_data['status'] == 'success'
    mock_stop_recording.assert_called_once()

    # 4. Verificar o estado (deve estar parado)
    note_response_stopped = client.get('/get_note')
    note_data_stopped = note_response_stopped.get_json()
    assert note_data_stopped['status'] == 'stopped'


def test_get_note_returns_predicted_note(client, mocker):
    """
    Testa se o endpoint /get_note retorna a nota que foi 'detectada'.
    """
    # Mock para simular que o reconhecimento está ativo
    mocker.patch('app.prediction_active', True)
    # Define a nota 'detectada'
    mocker.patch('app.current_note', 'C_Major')

    response = client.get('/get_note')
    data = response.get_json()

    assert response.status_code == 200
    assert data['status'] == 'active'
    assert data['note'] == 'C_Major'