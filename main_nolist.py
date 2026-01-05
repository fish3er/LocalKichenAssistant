import os
from tqdm import tqdm
from datetime import datetime
from config import Config
from data_utils import download_fruits_data, get_test_df
from models import MoondreamWrapper, LlavaWrapper
from evaluator import generate_report

def main():
    # 1. Zarządzanie folderami (Tylko na potrzeby wykresów)
    BASE_RESULTS_DIR = "/mnt/DyskDodatkowy/LAK_Rybak/wyniki_nolist"
    
    if not os.path.exists(BASE_RESULTS_DIR):
        os.makedirs(BASE_RESULTS_DIR)

    # Tworzymy folder dla konkretnego testu (tam trafią tylko wykresy)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, f"raport_{Config.MODEL_NAME}_{timestamp}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 2. Przygotowanie danych
    download_fruits_data(Config)
    df = get_test_df(Config)
    valid_classes = sorted(df['true_label'].unique())
    
    # 3. Wybór modelu
    vlm = MoondreamWrapper(Config) if Config.MODEL_NAME == "moondream" else LlavaWrapper(Config)
        
    # 4. Pętla testowa
    y_true, y_pred = [], []
    prompt = "Identify the fruit or vegetable in this image. Return only the name."
    
    print(f" Start testu {Config.MODEL_NAME}...")
    print(f" Raport zostanie zapisany w: {RESULTS_DIR}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            raw_answer = vlm.predict(row['path'], prompt)
            
            # Post-processing: Dopasowanie odpowiedzi do znanych klas
            prediction = "Unknown"
            clean_answer = raw_answer.lower().strip()
            
            for cls in valid_classes:
                if cls.lower() in clean_answer:
                    prediction = cls
                    break
            
            y_true.append(row['true_label'])
            y_pred.append(prediction)

            # --- USUNIĘTO shutil.copy ---
            # Zdjęcia nie są już kopiowane na dysk.

        except Exception as e:
            print(f"Błąd dla {row['path']}: {e}")

    # 5. Generowanie raportu (zapisuje macierz_pomylek.png i accuracy_klas.png)
    generate_report(y_true, y_pred, valid_classes, RESULTS_DIR)
    
      print(f" Gotowe. Wyniki (wykresy) znajdziesz w: {RESULTS_DIR}")

if __name__ == "__main__":
    main()