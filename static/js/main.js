document.addEventListener('DOMContentLoaded', async () => {
    const audioDeviceSelect = document.getElementById('audio-device');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const detectedNoteDisplay = document.getElementById('detected-note');

    let pollingIntervalId = null; // Para controlar o intervalo de polling

    // Função para listar dispositivos de áudio
    async function listAudioDevices() {
        try {
            const response = await fetch('/list_audio_devices');
            const devices = await response.json();
            audioDeviceSelect.innerHTML = ''; // Limpa as opções existentes

            if (devices.length === 0) {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'Nenhum dispositivo de entrada encontrado';
                audioDeviceSelect.appendChild(option);
                startBtn.disabled = true; // Desabilita o botão se não houver dispositivos
                return;
            }

            // Adiciona a opção "Padrão" primeiro
            const defaultOption = document.createElement('option');
            defaultOption.value = ''; // Valor vazio para dispositivo padrão
            defaultOption.textContent = 'Dispositivo Padrão do Sistema';
            audioDeviceSelect.appendChild(defaultOption);

            devices.forEach(device => {
                // Apenas dispositivos de entrada e que não sejam o "default" (já adicionado)
                if (device.max_input_channels > 0 && !device.name.toLowerCase().includes("default")) {
                    const option = document.createElement('option');
                    option.value = device.id;
                    option.textContent = device.name;
                    audioDeviceSelect.appendChild(option);
                }
            });
            startBtn.disabled = false; // Habilita o botão se houver dispositivos
        } catch (error) {
            console.error('Erro ao listar dispositivos de áudio:', error);
            detectedNoteDisplay.textContent = 'Erro ao carregar dispositivos.';
            startBtn.disabled = true;
        }
    }

    // Função para iniciar o reconhecimento
    startBtn.addEventListener('click', async () => {
        const selectedDeviceId = audioDeviceSelect.value; // Pode ser vazio para o padrão
        
        startBtn.disabled = true;
        stopBtn.disabled = false;
        audioDeviceSelect.disabled = true;
        detectedNoteDisplay.textContent = 'Iniciando...';

        try {
            const response = await fetch('/start_recognition', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ device_id: selectedDeviceId })
            });
            const data = await response.json();
            if (data.status === 'success' || data.status === 'warning') {
                detectedNoteDisplay.textContent = 'Ouvindo...';
                // Iniciar o loop de polling para obter os acordes
                startPollingForNotes();
            } else {
                detectedNoteDisplay.textContent = 'Erro ao iniciar: ' + data.message;
                startBtn.disabled = false;
                stopBtn.disabled = true;
                audioDeviceSelect.disabled = false;
            }
        } catch (error) {
            console.error('Erro ao iniciar o reconhecimento:', error);
            detectedNoteDisplay.textContent = 'Erro de comunicação com o servidor.';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            audioDeviceSelect.disabled = false;
        }
    });

    // Função para parar o reconhecimento
    stopBtn.addEventListener('click', async () => {
        // Limpar o intervalo de polling imediatamente
        stopPollingForNotes();

        startBtn.disabled = false;
        stopBtn.disabled = true;
        audioDeviceSelect.disabled = false;
        detectedNoteDisplay.textContent = 'Parando...';

        try {
            const response = await fetch('/stop_recognition', {
                method: 'POST'
            });
            const data = await response.json();
            if (data.status === 'success') {
                detectedNoteDisplay.textContent = 'Reconhecimento parado.';
            } else {
                detectedNoteDisplay.textContent = 'Erro ao parar: ' + data.message;
            }
        } catch (error) {
            console.error('Erro ao parar o reconhecimento:', error);
            detectedNoteDisplay.textContent = 'Erro de comunicação com o servidor.';
        }
    });

    // Inicia o loop de polling para obter os acordes detectados
    function startPollingForNotes() {
        if (pollingIntervalId) {
            clearInterval(pollingIntervalId); // Limpa qualquer polling anterior
        }
        pollingIntervalId = setInterval(async () => {
            try {
                const response = await fetch('/get_note'); // Rota ainda chamada get_note por conveniência
                const data = await response.json();
                if (data.status === 'stopped') {
                    // Se o servidor indicar que parou, encerra o polling
                    stopPollingForNotes();
                    detectedNoteDisplay.textContent = 'Reconhecimento parado.';
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    audioDeviceSelect.disabled = false;
                } else if (data.note) {
                    // Aqui você pode adicionar lógica para formatar o nome do acorde se quiser (ex: C_major -> C major)
                    detectedNoteDisplay.textContent = data.note.replace(/_/g, ' '); // Substitui underscores por espaços
                }
            } catch (error) {
                console.error('Erro ao obter acorde:', error);
                detectedNoteDisplay.textContent = 'Erro ao obter acorde. Reinicie.';
                stopPollingForNotes(); // Parar o polling em caso de erro
                startBtn.disabled = false;
                stopBtn.disabled = true;
                audioDeviceSelect.disabled = false;
            }
        }, 150); // Polling a cada 150ms
    }

    // Para o loop de polling
    function stopPollingForNotes() {
        if (pollingIntervalId) {
            clearInterval(pollingIntervalId);
            pollingIntervalId = null;
        }
    }

    // Inicializar ao carregar a página
    listAudioDevices();
});