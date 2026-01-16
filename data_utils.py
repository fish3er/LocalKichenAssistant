import os
import pandas as pd
import subprocess

def download_fruits_data(config):
    """Pobiera dane z Kaggle jeśli folder główny wersji nie istnieje."""
    target_dir = os.path.join(config.DATA_DIR, config.DATASET_VERSION)
    
    if not os.path.exists(target_dir):
        print(f" Pobieranie danych {config.KAGGLE_DATASET} do {target_dir}...")
        if not os.path.exists(config.DATA_DIR):
            os.makedirs(config.DATA_DIR)
            
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", config.KAGGLE_DATASET, 
            "-p", target_dir, 
            "--unzip"
        ])
        print(" Pobieranie zakończone.")
    else:
        print(f" Folder {target_dir} już istnieje. Pomijam pobieranie.")

def find_images_folder(start_path):
    """
    Rekurencyjnie szuka folderu zawierającego zdjęcia testowe.
    Priorytet: folder o nazwie 'Test', a jeśli brak to 'Validation'.
    """
    print(f" Szukam folderu z danymi w: {start_path} ...")
    
    candidate = None
    
    for root, dirs, files in os.walk(start_path):
        # Szukamy folderu Test
        if "Test" in dirs:
            return os.path.join(root, "Test")
        # Szukamy folderu Validation (częste w wersji original-size)
        if "Validation" in dirs:
            candidate = os.path.join(root, "Validation")
            
        # Jeśli znaleźliśmy Validation, ale szukamy dalej Testu (bo Test jest lepszy),
        # to nie przerywamy od razu, chyba że zejdziemy za głęboko.
        # Dla uproszczenia: jeśli znajdziemy Test to zwracamy od razu.
        
    # Jeśli przeszliśmy wszystko i nie ma Test, ale było Validation, zwracamy Validation
    if candidate:
        print(f" Nie znaleziono folderu 'Test', używam 'Validation'.")
        return candidate

    return None

def get_test_df(config):
    """Tworzy DataFrame, automatycznie znajdując ścieżkę do zdjęć."""
    
    # Punkt startowy poszukiwań: data/NAZWA_WERSJI
    base_search_path = os.path.join(config.DATA_DIR, config.DATASET_VERSION)
    
    # Automatyczne szukanie właściwego podfolderu
    final_path = find_images_folder(base_search_path)
    
    if not final_path:
        # Ostatnia deska ratunku - może użytkownik nie ma podfolderów wersji?
        # Sprawdźmy w samym 'data'
        final_path = find_images_folder(config.DATA_DIR)

    if not final_path:
        raise FileNotFoundError(
            f"Nie udało się znaleźć folderu 'Test' ani 'Validation' wewnątrz {base_search_path}. "
            "Sprawdź czy dane zostały poprawnie pobrane i rozpakowane."
        )

    print(f" Wczytywanie zdjęć z: {final_path}")
    
    data = []
    # Sortujemy foldery
    if not os.path.exists(final_path):
         raise FileNotFoundError(f"Ścieżka nie istnieje: {final_path}")

    folders = sorted([d for d in os.listdir(final_path) if os.path.isdir(os.path.join(final_path, d))])
    
    for folder_name in folders:
        folder_path = os.path.join(final_path, folder_name)
        
        # Logika czyszczenia nazw (usuwanie cyfr, znaków specjalnych)
        # np. "apple_hit_1" -> "apple", "Banana" -> "banana"
        clean_label = folder_name.replace("-", " ").replace("_", " ").lower()
        clean_label = ''.join([i for i in clean_label if not i.isdigit()]).strip()
        # Bierzemy pierwsze słowo jako główną kategorię (opcjonalne, zależy od potrzeb)
        clean_label = clean_label.split()[0] 
        
        # Pobieranie zdjęć
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files = sorted(files)
        
        for f in files[:config.IMAGES_PER_CLASS]:
            data.append({
                "path": os.path.join(folder_path, f), 
                "true_label": clean_label
            })
            
    if not data:
        raise ValueError("Znaleziono folder, ale jest pusty lub nie zawiera zdjęć (jpg/png).")

    return pd.DataFrame(data)