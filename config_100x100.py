import torch

class Config:
    # Wybór modelu: "moondream" lub "llava"
    MODEL_NAME = "llava" 
    
    # Dane
    DATA_DIR = "data"
    DATASET_VERSION = "fruits-360_100x100"  # Dla małych obrazów 100x100
    KAGGLE_DATASET = "moltean/fruits"
    IMAGES_PER_CLASS = 10
    
    # Sprzęt
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAP_LOCATION = "auto" # dla LLaVA (bitsandbytes)