
# VIOLÃO AI

Este projeto treina um dataset de audios de acordes musicais com o metodo CNN e disponibiliza uma API Flask para visualizar uma interface onde o usuário podera ver o recenhecimento do acorde tocado.

---

## Requisitos

- Python 3.11

---

## 1. Instalação

### Passo 1: Clone o repositório

```bash
git clone https://github.com/imxder/shopee-product-matching
cd VIOLAO_AI
```

### Passo 2: Crie e ative o ambiente virtual

No Windows:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Passo 3: Instale as dependências

```bash
pip install -r requirements.txt
```

## 2. Baixe o DATASET.

- Execute o codigo de `baixar_bataset.py`.

```bash
python baixar_bataset.py
```

Este script deve:

- Baixar a pasta do dataset do Google Drive descomprimir.
- Descomprimir pasta `dataset.zip`.

## 3. Treinar o modelo CNN 1D

Ao clonar o repositório já vem o modelo treinado,
para treinar novamente basta apagar o conteúdo da pasta `trained_model`.

Execute o treinamento do modelo:

```bash
python train_model.py
```

O que este script faz:

- Carrega os audios da pasta `dataset`.
- Treina o modelo CNN.

## 4. Rodar a API Flask

Para iniciar a API Flask que exibe os produtos e recomendações:

```bash
python app.py
```

Acesse no navegador:

```
http://localhost:5000
```

Você verá a interface web com imagens dos produtos, recomendados pelo modelo.

---

## Estrutura dos arquivos
```
/
├── app.py                     # API Flask (Aplicação Principal)
├── baixar_dataset.py          # Script para baixar o dataset (via gdown)
├── train_model.py             # Script para treinar modelo (CNN 1D)
├── audio_catpure.py           # Script para captura de áudio do usuário (Microfone ou Interface)
|
├── requirements.txt           # Dependências do projeto
|
├── static/                    # Arquivos estáticos (CSS, JS)
│   └── css/                   
│       └── style.css│         # Arquivo CSS para estilização  
│   └── js/                   
│       └── main.js            # Arquivo javascript
│
├── templates/                 # Templates HTML do Flask
│   └── index.html             # Página principal da aplicação
|
└── trained_model/             # Pasta com modelos já treinados
    ├── chord_recognizer_cnn_model.h5
    ├── label_encoder_chords.joblib
    └── scaler_chords.joblib
```
