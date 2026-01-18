import pandas as pd
from tqdm import tqdm
import os

# Importy narzędzi
from data_utils import download_fruits_data, get_train_test_dfs
from evaluator import generate_report
from config_dino import DinoConfig
from dino_wrapper import DinoWrapper

def main():
    # --- 1. PRZYGOTOWANIE DANYCH ---
    print("\n=== KROK 1: POBIERANIE I PRZYGOTOWANIE DANYCH ===")
    download_fruits_data(DinoConfig)
    
    # Pobieramy dwa osobne zestawy danych
    # train_df -> z folderu Training (służy do budowy wzorca)
    # test_df  -> z folderu Test (służy do weryfikacji)
    train_df, test_df = get_train_test_dfs(DinoConfig)
    
    all_classes = sorted(train_df['true_label'].unique())
    print(f" -> Znaleziono {len(all_classes)} unikalnych klas.")
    print(f" -> Liczba zdjęć do budowy wzorców (Training): {len(train_df)}")
    print(f" -> Liczba zdjęć do testowania (Test):        {len(test_df)}")

    # --- 2. PRZYGOTOWANIE SŁOWNIKA WZORCÓW ---
    # DinoWrapper wymaga formatu: { "nazwa_klasy": ["sciezka1.jpg", "sciezka2.jpg", ...] }
    ref_paths_dict = {}
    
    for cls in all_classes:
        # Wybieramy ścieżki tylko dla danej klasy ze zbioru TRAINING
        paths = train_df[train_df['true_label'] == cls]['path'].tolist()
        if paths:
            ref_paths_dict[cls] = paths

    # --- 3. INICJALIZACJA MODELU I BUDOWA PROTOTYPÓW ---
    print("\n=== KROK 2: INICJALIZACJA MODELU DINOv2 ===")
    model = DinoWrapper(device=DinoConfig.DEVICE)
    
    # Model "uczy się" (oblicza średnie wektory) tylko na danych treningowych
    model.fit_prototypes(ref_paths_dict)

    # --- 4. TESTOWANIE ---
    print(f"\n=== KROK 3: TESTOWANIE ({len(test_df)} zdjęć) ===")
    
    y_true = []       # Prawdziwe etykiety
    y_pred = []       # Przewidziane etykiety
    y_scores = []     # Wyniki pewności (podobieństwo cosinusowe)
    all_embeddings = [] # Wektory do wizualizacji t-SNE
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        true_lbl = row['true_label']
        path = row['path']
        
        # Predykcja na zdjęciu testowym
        # Zwraca: przewidzianą klasę, wynik punktowy (0-1) i wektor cech
        pred_lbl, score, emb_vector = model.predict(path)
        
        if emb_vector is not None:
            y_true.append(true_lbl)
            y_pred.append(pred_lbl)
            y_scores.append(score)
            all_embeddings.append(emb_vector)
        else:
            print(f"Błąd przetwarzania pliku: {path}")

    # --- 5. RAPORTOWANIE ---
    print("\n=== KROK 4: GENEROWANIE RAPORTU I WYKRESÓW ===")
    
    # Wywołanie zaktualizowanej funkcji generate_report
    # Argumenty opcjonalne (y_scores, embeddings) przekazujemy po nazwie
    generate_report(
        y_true, 
        y_pred, 
        all_classes,          # Lista wszystkich możliwych klas (do macierzy pomyłek)
        y_scores=y_scores,    # Potrzebne do wykresów Gaussa
        embeddings=all_embeddings # Potrzebne do wykresów t-SNE
    )

if __name__ == "__main__":
    main()