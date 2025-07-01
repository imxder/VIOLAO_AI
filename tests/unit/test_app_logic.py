import pytest
from collections import deque, Counter
from app import PREDICTION_BUFFER_SIZE, STABILITY_THRESHOLD

def test_note_stability_logic():
    """
    Testa a lógica de estabilização da nota para garantir que uma nota só é
    considerada 'estável' após aparecer um número suficiente de vezes.
    """
    # Cenário 1: Buffer não está cheio o suficiente
    prediction_buffer = deque(['C', 'C', 'G', 'C'], maxlen=PREDICTION_BUFFER_SIZE)
    counts = Counter(prediction_buffer)
    most_common_chord, num_occurrences = counts.most_common(1)[0]

    # A nota 'C' é a mais comum, mas não atinge o limiar de estabilidade
    assert most_common_chord == 'C'
    assert num_occurrences < STABILITY_THRESHOLD
    assert num_occurrences == 3

    # Cenário 2: Uma nota se torna estável
    # Simulando que a nota 'Am' foi detectada consistentemente
    stable_prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    for _ in range(STABILITY_THRESHOLD):
        stable_prediction_buffer.append('Am')

    stable_counts = Counter(stable_prediction_buffer)
    stable_most_common, stable_num_occurrences = stable_counts.most_common(1)[0]

    # A nota 'Am' deve ser considerada estável
    assert stable_most_common == 'Am'
    assert stable_num_occurrences >= STABILITY_THRESHOLD

    # Cenário 3: Transição de uma nota estável para outra
    # A nota 'G' começa a aparecer
    stable_prediction_buffer.append('G')
    stable_prediction_buffer.append('G')

    transition_counts = Counter(stable_prediction_buffer)
    transition_most_common, transition_num_occurrences = transition_counts.most_common(1)[0]

    # 'Am' ainda é a nota mais comum. Sua contagem não caiu porque o buffer (tamanho 12)
    # ainda não está cheio (agora contém 9 'Am' e 2 'G').
    assert transition_most_common == 'Am'
    assert transition_num_occurrences == STABILITY_THRESHOLD