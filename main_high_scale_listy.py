import sys
import os
from tqdm import tqdm

# Dynamic import of config
config_name = sys.argv[1] if len(sys.argv) > 1 else 'config'
config_module = __import__(config_name)
Config = config_module.Config

from data_utils import download_fruits_data, get_test_df
from models import MoondreamWrapper, LlavaWrapper
from evaluator import generate_report 

def main():
    # 1. Przygotowanie danych
    download_fruits_data(Config)
    df = get_test_df(Config)
    valid_classes = sorted(df['true_label'].unique())
    class_list_str = ", ".join([c.upper() for c in valid_classes])
    
    # 2. Wybór modelu
    vlm = MoondreamWrapper(Config) if Config.MODEL_NAME == "moondream" else LlavaWrapper(Config)
        
    # 3. Pętla testowa
    y_true, y_pred = [], []
    prompt = f"Classify this image. Select the best category from: {class_list_str}. Return only the name."
    
    print(f" Start testu {Config.MODEL_NAME}...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            raw_answer = vlm.predict(row['path'], prompt)
            
            prediction = "Mismatch"
            # Szukanie klasy w odpowiedzi modelu
            for cls in valid_classes:
                if cls.lower() in raw_answer.lower():
                    prediction = cls
                    break
            
            y_true.append(row['true_label'])
            y_pred.append(prediction)

            # BRAK ZAPISYWANIA ZDJĘĆ

        except Exception as e:
            print(f"Błąd dla {row['path']}: {e}")

    # 4. Wywołanie raportu
    # Funkcja obliczy statystyki i wyświetli je w konsoli.
    # (Pamiętaj, że funkcja generate_report, którą podałeś wcześniej, 
    # sama w sobie zawiera kod zapisujący wykresy PNG na dysk).
    generate_report(y_true, y_pred, valid_classes)

if __name__ == "__main__":
    main()