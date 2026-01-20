# Dokumentacja techniczna projektu

## Opis plików

###  Konfiguracja (`config_*.py`)
- **`config_original.py`**: Główne ustawienia projektu (ścieżki, hiperparametry) dla oryginalnej skali obrazów.
- **`config_100x100.py`**: Konfiguracja zoptymalizowana pod niską rozdzielczość (100x100 px), przyspieszająca procesy testowe.
- **`config_dino.py`**: Specyficzne parametry dla modelu DINO (np. rozmiary patchy, progi detekcji).

### Logika modeli i AI
- **`models.py`**: Definicje architektur sieci neuronowych wykorzystywanych w projekcie.
- **`dino_wrapper.py`**: Klasa pomocnicza (wrapper) do obsługi modelu DINO – odpowiada za ładowanie wag, przetwarzanie wstępne i ekstrakcję cech.
- **`run_dino_experiment.py`**: Skrypt dedykowany do przeprowadzania ustandaryzowanych eksperymentów z modelem DINO.

### Skrypty uruchomieniowe (`main_*.py`)
- **`main_list.py`**: Główny skrypt obsługujący dane wejściowe w formie list obiektów.
- **`main_nolist.py`**: Wersja skryptu pracująca na pojedynczych obiektach lub strumieniu bez struktur listowych.
- **`main_high_scale_listy.py`**: Przetwarzanie obrazów w wysokiej rozdzielczości z obsługą list.
- **`main_high_scale_no_list.py`**: Przetwarzanie obrazów w wysokiej rozdzielczości bez list.

### Narzędzia i dane
- **`data_utils.py`**: Funkcje pomocnicze do ładowania obrazów, augmentacji, transformacji danych oraz obsługi datasetów.
- **`evaluator.py`**: Narzędzie do ewaluacji modeli – oblicza metryki (np. Accuracy, mAP) i generuje raporty z wyników.

---

## Instalacja

1. **Klonowanie repozytorium:**
   ```bash
   git clone https://github.com/fish3er/LocalKichenAssistant.git
   cd LocalKichenAssistant
   ```

2. **Instalacja zależności:**
   Upewnij się, że masz zainstalowanego Pythona (zalecany 3.9+), a następnie wykonaj:
   ```bash
   pip install -r requirements.txt
   ```

---

## Sposób uruchomienia

### Wykonanie głównego algorytmu:
Wybierz skrypt odpowiadający Twoim potrzebom (np. wersja z listami):
```bash
python main_list.py
```

### Przeprowadzenie eksperymentu:
Jeśli chcesz przetestować model DINO na swoich danych:
```bash
python run_dino_experiment.py
```

