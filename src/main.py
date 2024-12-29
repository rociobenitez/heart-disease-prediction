from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_training import train_random_forest, train_logistic_regression, train_knn
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# Cargar los datos
data = pd.read_csv("data/heart-disease.csv")

# Dividir en X e y
X = data.drop("target", axis=1)
y = data["target"]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelos
rf_model, rf_params = train_random_forest(X_train, y_train)
log_reg_model, log_reg_params = train_logistic_regression(X_train_scaled, y_train)
knn_model, knn_params = train_knn(X_train_scaled, y_train)

# Evaluar modelos
for model, name in zip([rf_model, log_reg_model, knn_model], ["Random Forest", "Logistic Regression", "KNN"]):
    if name in ['Logistic Regression', 'KNN']:
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"Resultados para {name}:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.2f}\n")
