document.addEventListener('DOMContentLoaded', async () => {
    const audioDeviceSelect = document.getElementById('audio-device');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const detectedNoteDisplay = document.getElementById('detected-note');
    const resultContainer = document.querySelector('.result'); 

    let pollingIntervalId = null;

    async function listAudioDevices() {
        try {
            const response = await fetch('/list_audio_devices');
            const devices = await response.json();
            audioDeviceSelect.innerHTML = '';

            if (devices.length === 0) {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'Nenhum dispositivo encontrado';
                audioDeviceSelect.appendChild(option);
                startBtn.disabled = true;
                return;
            }

            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Dispositivo Padrão do Sistema';
            audioDeviceSelect.appendChild(defaultOption);

            devices.forEach(device => {
                if (device.max_input_channels > 0 && !device.name.toLowerCase().includes("default")) {
                    const option = document.createElement('option');
                    option.value = device.id;
                    option.textContent = device.name;
                    audioDeviceSelect.appendChild(option);
                }
            });
            startBtn.disabled = false;
        } catch (error) {
            console.error('Erro ao listar dispositivos de áudio:', error);
            detectedNoteDisplay.textContent = 'Erro';
            startBtn.disabled = true;
        }
    }

    startBtn.addEventListener('click', async () => {
        const selectedDeviceId = audioDeviceSelect.value;
        
        startBtn.disabled = true;
        stopBtn.disabled = false;
        audioDeviceSelect.disabled = true;
        detectedNoteDisplay.textContent = 'Iniciando...';
        resultContainer.classList.remove('is-listening'); // Garante estado limpo

        try {
            const response = await fetch('/start_recognition', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ device_id: selectedDeviceId })
            });
            const data = await response.json();
            if (data.status === 'success' || data.status === 'warning') {
                detectedNoteDisplay.textContent = 'Ouvindo...';
                resultContainer.classList.add('is-listening'); 
                startPollingForNotes();
            } else {
                detectedNoteDisplay.textContent = 'Erro!';
                stopRecognitionCleanup();
            }
        } catch (error) {
            console.error('Erro ao iniciar:', error);
            detectedNoteDisplay.textContent = 'Erro!';
            stopRecognitionCleanup();
        }
    });

    stopBtn.addEventListener('click', async () => {
        stopPollingForNotes();
        stopRecognitionCleanup();
        detectedNoteDisplay.textContent = 'Parado';

        try {
            await fetch('/stop_recognition', { method: 'POST' });
        } catch (error) {
            console.error('Erro ao parar:', error);
            detectedNoteDisplay.textContent = 'Erro!';
        }
    });

    function startPollingForNotes() {
        if (pollingIntervalId) clearInterval(pollingIntervalId);

        pollingIntervalId = setInterval(async () => {
            try {
                const response = await fetch('/get_note');
                const data = await response.json();

                if (data.status === 'stopped') {
                    stopPollingForNotes();
                    stopRecognitionCleanup();
                    detectedNoteDisplay.textContent = 'Parado';
                } else if (data.note) {
                    resultContainer.classList.remove('is-listening'); 
                    detectedNoteDisplay.textContent = data.note.replace(/_/g, ' ');
                } else {
                    if (!resultContainer.classList.contains('is-listening')) {
                        resultContainer.classList.add('is-listening');
                        detectedNoteDisplay.textContent = 'Ouvindo...';
                    }
                }
            } catch (error) {
                console.error('Erro no polling:', error);
                detectedNoteDisplay.textContent = 'Erro!';
                stopPollingForNotes();
                stopRecognitionCleanup();
            }
        }, 200); 
    }

    function stopPollingForNotes() {
        if (pollingIntervalId) {
            clearInterval(pollingIntervalId);
            pollingIntervalId = null;
        }
    }

    function stopRecognitionCleanup() {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        audioDeviceSelect.disabled = false;
        resultContainer.classList.remove('is-listening'); 
    }
    
    listAudioDevices();
    detectedNoteDisplay.textContent = '--';
});