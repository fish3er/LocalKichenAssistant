import os
from tqdm import tqdm
from config_original import Config
from data_utils import download_fruits_data, get_test_df
# Musimy zaimportować klasę Qwen2VLWrapper
from models import MoondreamWrapper, LlavaWrapper, Qwen2VLWrapper
from evaluator import generate_report 

def main():
   # 1. Przygotowanie danych (bez zmian)
    download_fruits_data(Config)
    df = get_test_df(Config)
    valid_classes = sorted(df['true_label'].unique())
    class_list_str = ", ".join([c.upper() for c in valid_classes])
    
    # 2. Wybór modelu (Zaktualizowano pod Qwen)
    print(f"Wybrano model: {Config.MODEL_NAME}")
    
    if Config.MODEL_NAME == "moondream":
        vlm = MoondreamWrapper(Config)
    elif Config.MODEL_NAME == "llava":
        vlm = LlavaWrapper(Config)
    elif Config.MODEL_NAME == "qwen":
        vlm = Qwen2VLWrapper(Config)
    else:
        # Fallback - jeśli nazwa jest inna, domyślnie użyjemy Qwena lub rzucimy błąd
        print(f"Nieznany model '{Config.MODEL_NAME}', używam Qwen jako domyślny.")
        vlm = Qwen2VLWrapper(Config)
        
    
    if Config.MODEL_NAME == "moondream":
        vlm = MoondreamWrapper(Config)
    elif Config.MODEL_NAME == "llava":
        vlm = LlavaWrapper(Config)
    elif Config.MODEL_NAME == "qwen":
        vlm = Qwen2VLWrapper(Config)
    else:
        raise ValueError(f"Nieznany model w konfiguracji: {Config.MODEL_NAME}")
        
    # 3. Pętla testowa
    y_true, y_pred = [], []
    prompt = f"Classify this image. Return only the name."
    
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