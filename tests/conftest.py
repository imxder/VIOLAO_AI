import pytest
import sys
import os

# Adiciona o diretório raiz do projeto (a pasta acima de 'tests') ao PythonPath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app

@pytest.fixture
def app():
    """Cria e configura uma nova instância do aplicativo para cada teste."""
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app

@pytest.fixture
def client(app):
    """Um cliente de teste para o aplicativo."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Um executor de teste para a CLI do aplicativo."""
    return app.test_cli_runner()