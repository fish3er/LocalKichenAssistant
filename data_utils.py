import os
import pandas as pd
import subprocess

def download_fruits_data(config):
    """Pobiera dane z Kaggle jeśli ich nie ma."""
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
        print("📥 Pobieranie danych z Kaggle...")
        subprocess.run(["kaggle", "datasets", "download", "-d", config.KAGGLE_DATASET, "-p", config.DATA_DIR, "--unzip"])

def get_test_df(config):
    """Tworzy DataFrame ze ścieżkami i oczyszczonymi etykietami."""
    # Ścieżka może się różnić zależnie od tego jak kaggle wypakuje pliki
    path = os.path.join(config.DATA_DIR, "fruits-360_100x100", "fruits-360", "Test")
    
    data = []
    for folder_name in sorted(os.listdir(path)):
        folder_path = os.path.join(path, folder_name)
        if not os.path.isdir(folder_path): continue
        
        # Twoja logika czyszczenia nazw (np. Apple Red 1 -> apple)
        clean_label = folder_name.replace("_", " ").split()[0].lower()
        clean_label = ''.join([i for i in clean_label if not i.isdigit()])
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files[:config.IMAGES_PER_CLASS]:
            data.append({"path": os.path.join(folder_path, f), "true_label": clean_label})
            
    return pd.DataFrame(data)