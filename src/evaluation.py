from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    """
    Evalúa un modelo en términos de métricas de clasificación.
    """
    y_preds = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "Accuracy": accuracy_score(y_test, y_preds),
        "Precision": precision_score(y_test, y_preds),
        "Recall": recall_score(y_test, y_preds),
        "F1-Score": f1_score(y_test, y_preds),
        "ROC-AUC": roc_auc_score(y_test, y_probs) if y_probs is not None else "N/A"
    }
    return metrics

def plot_confusion_matrix(model, X_test, y_test):
    """
    Dibuja la matriz de confusión.
    """
    y_preds = model.predict(X_test)
    cm = confusion_matrix(y_test, y_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Etiqueta predicha")
    plt.ylabel("Etiqueta verdadera")
    plt.title("Confusion Matrix")
    plt.show()
