import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def generate_report(y_true, y_pred, labels):
    # 1. Obliczenia
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    # 2. Konsola: Liczba klas, lista i wyniki
    print("\n" + "="*30)
    print(f"Liczba klas: {len(labels)}")  # <--- Dodano liczbę klas
    print(f"Lista klas: {', '.join(labels)}")
    print(f" Ogólne Accuracy: {acc:.2%}")
    print("-" * 30)
    
    class_accs = []
    for label in labels:
        c_acc = report[label]['recall']
        class_accs.append(c_acc)
        print(f"{label}: {c_acc:.2%}")

    # 3. WYKRES 1: Heatmapa (Tylko kolory, bez numerów)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, xticklabels=labels, yticklabels=labels, cmap='Reds')
    plt.xlabel("Przewidziane")
    plt.ylabel("Prawdziwe")
    plt.title(f"Macierz pomyłek (Total Acc: {acc:.2%})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("macierz_pomylek.png")
    plt.show()

    # 4. WYKRES 2: Accuracy dla każdej klasy + adnotacja ogólna
    plt.figure(figsize=(10, 16)) # Wysoki wykres ze względu na dużą liczbę klas
    colors = sns.color_palette("viridis", len(labels))
    plt.barh(labels, class_accs, color=colors)
    
    # Czerwona linia oznaczająca ogólne accuracy
    plt.axvline(x=acc, color='red', linestyle='--', linewidth=2, label=f'Ogólne Accuracy ({acc:.2%})')
    
    # Adnotacja z ogólnym wynikiem na środku wykresu
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
    plt.savefig("accuracy_klas.png")
    plt.show()
    
    print(f"\n Raport wygenerowany. Liczba klas: {len(labels)}")
    print("Zapisano pliki: 'macierz_pomylek.png' oraz 'accuracy_klas.png'")