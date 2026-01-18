import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE

def plot_tsne_separated(embeddings, labels, results_dir):
    # (Ten fragment kodu bez zmian - t-SNE separated)
    print("\n--> Generowanie t-SNE...")
    if isinstance(embeddings, list): embeddings = np.array(embeddings)
    embeddings = embeddings.squeeze()
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(embeddings)
    
    tsne_dir = os.path.join(results_dir, "tsne_pojedyncze")
    os.makedirs(tsne_dir, exist_ok=True)
    unique_classes = sorted(list(set(labels)))
    
    for target_cls in unique_classes:
        plt.figure(figsize=(10, 8))
        is_target = (np.array(labels) == target_cls)
        plt.scatter(tsne_results[~is_target, 0], tsne_results[~is_target, 1], c='lightgray', alpha=0.3, s=20)
        plt.scatter(tsne_results[is_target, 0], tsne_results[is_target, 1], c='red', alpha=0.9, s=40, edgecolors='black')
        plt.title(f"Klaster: {target_cls}")
        plt.savefig(os.path.join(tsne_dir, f"tsne_{target_cls.replace(' ', '_')}.png"))
        plt.close()

def plot_per_class_distributions(y_true, y_scores, labels, results_dir):
    # (Ten fragment kodu bez zmian - Gauss)
    print(f"\n--> Generowanie rozkładów...")
    dist_dir = os.path.join(results_dir, "distrybucje_klas")
    os.makedirs(dist_dir, exist_ok=True)
    y_true = np.array(y_true); y_scores = np.array(y_scores)
    for cls in labels:
        cls_scores = y_scores[y_true == cls]
        if len(cls_scores) < 2: continue
        plt.figure(figsize=(8, 6))
        sns.histplot(cls_scores, kde=True, bins=15, color='skyblue', edgecolor='black')
        plt.title(f"Pewność: {cls}")
        plt.savefig(os.path.join(dist_dir, f"dist_{cls.replace(' ', '_')}.png"))
        plt.close()

# --- ZMODYFIKOWANA FUNKCJA GŁÓWNA ---
# Teraz y_scores i embeddings są OPCJONALNE i na końcu
def generate_report(y_true, y_pred, labels, y_scores=None, embeddings=None):
    results_dir = "wyniki"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Metryki
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*30)
    print(f" Ogólne Accuracy: {acc:.2%}")
    print("-" * 30)

    # 2. Wykresy standardowe (dla wszystkich modeli)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, xticklabels=labels, yticklabels=labels, cmap='Reds')
    plt.title(f"Macierz pomyłek (Acc: {acc:.2%})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "macierz_pomylek.png"))
    plt.close()

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    plt.figure(figsize=(10, 16))
    class_accs = [report[label]['recall'] for label in labels]
    plt.barh(labels, class_accs, color=sns.color_palette("viridis", len(labels)))
    plt.axvline(x=acc, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_klas.png"))
    plt.close()

    # 3. Wykresy zaawansowane (Tylko dla DINO)
    if y_scores is not None:
        plot_per_class_distributions(y_true, y_scores, labels, results_dir)

    if embeddings is not None:
        try:
            plot_tsne_separated(embeddings, y_true, results_dir)
        except Exception as e:
            print(f"Błąd t-SNE: {e}")

    print(f"\n Wyniki w: {os.path.abspath(results_dir)}")