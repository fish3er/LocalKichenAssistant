import os
import shutil
from tqdm import tqdm
from config import Config
from data_utils import download_fruits_data, get_test_df
from models import MoondreamWrapper, LlavaWrapper
from evaluator import generate_report # Importujemy funkcję z drugiego pliku

def main():
    # ŚCIEŻKA DO WYNIKÓW
    RESULTS_DIR = "/mnt/DyskDodatkowy/LAK_Rybak/wyniki"
    
    # Czyszczenie/tworzenie folderu na starcie
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

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
    
    print(f"🚀 Start testu {Config.MODEL_NAME}...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            raw_answer = vlm.predict(row['path'], prompt)
            
            prediction = "Mismatch"
            for cls in valid_classes:
                if cls in raw_answer.lower():
                    prediction = cls
                    break
            
            y_true.append(row['true_label'])
            y_pred.append(prediction)

            # Zapis zdjęcia do folderu wyniki
            file_ext = os.path.splitext(row['path'])[1]
            filename = f"{idx}_TRUE_{row['true_label']}_PRED_{prediction}{file_ext}"
            shutil.copy(row['path'], os.path.join(RESULTS_DIR, filename))

        except Exception as e:
            print(f"Błąd dla {row['path']}: {e}")

    # 4. Wywołanie raportu z drugiego pliku
    generate_report(y_true, y_pred, valid_classes)

if __name__ == "__main__":
    main()