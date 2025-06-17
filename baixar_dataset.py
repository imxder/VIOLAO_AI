import gdown
import zipfile
import os
import time 

file_id = '1ZeyJx1pJhJ21MfeMvty34Rn5c-e3QXyF'
zip_filename = 'dataset.zip'
dest_folder = '.'
download_url = f'https://drive.google.com/uc?id={file_id}'

print(f"Baixando {zip_filename}...")

try:
    gdown.download(download_url, zip_filename, quiet=False)
    print(f"{zip_filename} baixado com sucesso!")

    print("Aguardando 2 segundos antes de extrair...")
    time.sleep(2) 
    print(f"\nExtraindo {zip_filename} para '{os.path.abspath(dest_folder)}'...")

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    print("Arquivos extraídos com sucesso!")

    time.sleep(1)

    os.remove(zip_filename)
    print(f"{zip_filename} removido.")

except Exception as e:
    print(f"ERRO: Ocorreu um problema durante o download ou extração: {e}")

print("\nProcesso concluído.")