import os
import matplotlib
# USTAWIENIE BACKENDU przed importem pyplot (ważne dla terminala!)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def generate_report(y_true, y_pred, labels):
    # Ścieżka do folderu z wynikami
    results_dir = "/mnt/DyskDodatkowy/LAK_Rybak/wyniki"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Obliczenia
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    # 2. Konsola
    print("\n" + "="*30)
    print(f"Liczba klas: {len(labels)}")
    print(f" Ogólne Accuracy: {acc:.2%}")
    print("-" * 30)

    # 3. WYKRES 1: Macierz pomyłek
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, xticklabels=labels, yticklabels=labels, cmap='Reds')
    plt.xlabel("Przewidziane")
    plt.ylabel("Prawdziwe")
    plt.title(f"Macierz pomyłek (Total Acc: {acc:.2%})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Zapis i zamkniecie
    plt.savefig(os.path.join(results_dir, "macierz_pomylek.png"))
    plt.close() 

    # 4. WYKRES 2: Accuracy dla każdej klasy
    plt.figure(figsize=(10, 16))
    class_accs = [report[label]['recall'] for label in labels]
    colors = sns.color_palette("viridis", len(labels))
    plt.barh(labels, class_accs, color=colors)
    
    plt.axvline(x=acc, color='red', linestyle='--', linewidth=2, label=f'Ogólne Accuracy ({acc:.2%})')
    plt.text(acc + 0.01, len(labels)/2, f'WYNIK CAŁKOWITY: {acc:.2%}', 
             color='red', fontweight='bold', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'))

    plt.xlabel('Accuracy (0.0 - 1.0)')
    plt.ylabel('Klasy')
    plt.title(f'Accuracy dla każdej z {len(labels)} klas')
    plt.xlim(0, 1.05)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Zapis i zamkniecie
    plt.savefig(os.path.join(results_dir, "accuracy_klas.png"))
    plt.close() 
    
    print(f"\n Wykresy zapisano w folderze: {results_dir}")