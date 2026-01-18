import os
import pandas as pd
import subprocess

def download_fruits_data(config):
    """Pobiera dane z Kaggle."""
    target_dir = os.path.join(config.DATA_DIR, config.DATASET_VERSION)
    if not os.path.exists(target_dir):
        print(f" Pobieranie danych {config.KAGGLE_DATASET}...")
        if not os.path.exists(config.DATA_DIR):
            os.makedirs(config.DATA_DIR)
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", config.KAGGLE_DATASET, 
            "-p", target_dir, 
            "--unzip"
        ])
        print(" Pobieranie zakończone.")

def _clean_label_name(folder_name):
    """Logika czyszczenia nazw (wspólna dla obu metod)."""
    clean = folder_name.replace("-", " ").replace("_", " ").lower()
    clean = ''.join([i for i in clean if not i.isdigit()]).strip()
    clean = clean.split()[0] # Grupowanie (np. apple red 1 -> apple)
    return clean

def _scan_folder_for_images(base_path, max_images=None):
    """Pomocniczy skaner folderów."""
    data = []
    if not os.path.exists(base_path):
        return []

    subfolders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    for folder_name in subfolders:
        folder_path = os.path.join(base_path, folder_name)
        clean_label = _clean_label_name(folder_name)
        
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if max_images:
            files = files[:max_images]
            
        for f in files:
            data.append({
                "path": os.path.join(folder_path, f),
                "true_label": clean_label
            })
    return data

def _find_dir(base_path, dir_name):
    """Szuka folderu Training lub Test rekurencyjnie."""
    for root, dirs, files in os.walk(base_path):
        if dir_name in dirs:
            return os.path.join(root, dir_name)
    return None

# --- FUNKCJA DLA NOWEGO KODU (DINO) ---
def get_train_test_dfs(config):
    base_search_path = os.path.join(config.DATA_DIR, config.DATASET_VERSION)
    
    train_dir = _find_dir(base_search_path, "Training")
    test_dir = _find_dir(base_search_path, "Test")
            
    if not train_dir or not test_dir:
        raise FileNotFoundError("Nie znaleziono folderów Training/Test!")
        
    print(f" Indexing Training: {train_dir}")
    train_list = _scan_folder_for_images(train_dir, max_images=config.IMAGES_FOR_PROTOTYPE)
    
    print(f" Indexing Test: {test_dir}")
    test_list = _scan_folder_for_images(test_dir, max_images=config.IMAGES_FOR_TEST)
    
    return pd.DataFrame(train_list), pd.DataFrame(test_list)

# --- FUNKCJA DLA STAREGO KODU (Qwen/Moondream) ---
def get_test_df(config):
    """
    Przywrócona funkcja dla kompatybilności wstecznej.
    """
    base_search_path = os.path.join(config.DATA_DIR, config.DATASET_VERSION)
    
    # Próbujemy znaleźć folder Test, jak nie ma to Validation
    test_dir = _find_dir(base_search_path, "Test")
    if not test_dir:
        test_dir = _find_dir(base_search_path, "Validation")
        
    if not test_dir:
        # Ostateczność: szukaj w samym folderze wersji
        test_dir = base_search_path

    print(f" Wczytywanie zdjęć testowych z: {test_dir}")
    
    # Używamy config.IMAGES_PER_CLASS ze starego configu
    limit = getattr(config, 'IMAGES_PER_CLASS', 10)
    
    data_list = _scan_folder_for_images(test_dir, max_images=limit)
    
    if not data_list:
        raise ValueError("Nie znaleziono zdjęć!")
        
    return pd.DataFrame(data_list)