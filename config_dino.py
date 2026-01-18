import torch
from config_100x100 import Config as OldConfig

class DinoConfig:
    # Używamy ścieżek z podstawowej konfiguracji
    DATA_DIR = OldConfig.DATA_DIR
    DATASET_VERSION = OldConfig.DATASET_VERSION
    KAGGLE_DATASET = OldConfig.KAGGLE_DATASET
    
    # --- KONFIGURACJA ILOŚCI ZDJĘĆ ---
    
    # Ile zdjęć wziąć z folderu TRAINING (do budowy wzorca/prototypu)
    IMAGES_FOR_PROTOTYPE = 10 
    
    # Ile zdjęć wziąć z folderu TEST (do sprawdzenia dokładności)
    # Jeśli ustawisz None, weźmie wszystkie dostępne w folderze Test
    IMAGES_FOR_TEST = 10 
    
    # Sprzęt
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"