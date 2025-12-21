from config import Config
from data_utils import download_fruits_data, get_test_df
from models import MoondreamWrapper, LlavaWrapper
from evaluator import generate_report
from tqdm import tqdm

def main():
    # 1. Przygotowanie danych
    download_fruits_data(Config)
    df = get_test_df(Config)
    valid_classes = sorted(df['true_label'].unique())
    class_list_str = ", ".join([c.upper() for c in valid_classes])
    
    # 2. Inicjalizacja wybranego modelu
    if Config.MODEL_NAME == "moondream":
        vlm = MoondreamWrapper(Config)
    else:
        vlm = LlavaWrapper(Config)
        
    # 3. Pętla testowa
    y_true, y_pred = [], []
    prompt = f"Classify this image. Select the best category from: {class_list_str}. Return only the name."
    
    print(f"🚀 Start testu modelu {Config.MODEL_NAME}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            raw_answer = vlm.predict(row['path'], prompt)
            
            # Prosty post-processing (szukanie słowa kluczowego w odpowiedzi)
            prediction = "Mismatch"
            for cls in valid_classes:
                if cls in raw_answer.lower():
                    prediction = cls
                    break
            
            y_true.append(row['true_label'])
            y_pred.append(prediction)
        except Exception as e:
            print(f"Błąd dla {row['path']}: {e}")

    # 4. Raport końcowy
    generate_report(y_true, y_pred, valid_classes)

if __name__ == "__main__":
    main()