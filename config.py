import torch

class Config:
    # Wybór modelu: "moondream" lub "llava"
    MODEL_NAME = "llava" 
    
    # Dane
    DATA_DIR = "data"
    KAGGLE_DATASET = "moltean/fruits"
    IMAGES_PER_CLASS = 1
    
    # Sprzęt
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAP_LOCATION = "auto" # dla LLaVA (bitsandbytes)