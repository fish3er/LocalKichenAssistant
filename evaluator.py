import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def generate_report(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    print(f"📊 Accuracy: {acc:.2%}")
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Reds')
    plt.xlabel("Przewidziane")
    plt.ylabel("Prawdziwe")
    plt.title(f"Macierz pomyłek (Acc: {acc:.2%})")
    plt.savefig("wyniki_testu.png")
    plt.show()