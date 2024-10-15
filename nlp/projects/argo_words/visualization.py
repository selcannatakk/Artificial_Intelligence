import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def cm_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # Görselleştirme
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predict')
    plt.ylabel('Reel')
    plt.title('Confusion Matrix')
    plt.show()
