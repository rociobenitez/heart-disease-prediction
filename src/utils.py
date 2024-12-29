import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score

# ------------ EDA ------------

def plot_categorical_grid(data, columns, n_cols=3, figsize=(15, 10)):
    """
    Genera un grid de gráficos de barras para variables categóricas.

    Args:
        data (DataFrame): Dataset que contiene los datos.
        columns (list): Lista de nombres de las columnas categóricas.
        n_cols (int): Número de columnas en el grid (por defecto, 3).
        figsize (tuple): Tamaño del gráfico (por defecto, (15, 10)).
    """
    # Calcular número de filas
    n_rows = (len(columns) + n_cols - 1) // n_cols

    # Crear subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Asegurar que los ejes sean iterables

    # Iterar sobre las columnas categóricas
    for i, col in enumerate(columns):
        sns.countplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(f'Distribución de {col}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Conteo')

    # Eliminar gráficos vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar el layout
    plt.tight_layout()

    # Guardar y mostrar el gráfico
    plt.savefig('../assets/eda/categorical_countplot.png')
    plt.show()
    plt.close()

def plot_categorical_target_grid(data, columns, n_cols=3, figsize=(15, 10)):
    """
    Genera un grid de gráficos de barras para
    analizar cómo se relacionan las variables categóricas
    con la variable objetivo ('target').

    Args:
        data (DataFrame): Dataset que contiene los datos.
        columns (list): Lista de nombres de las columnas categóricas.
        n_cols (int): Número de columnas en el grid (por defecto, 3).
        figsize (tuple): Tamaño del gráfico (por defecto, (15, 10)).
    """
    # Calcular número de filas
    n_rows = (len(columns) + n_cols - 1) // n_cols

    # Crear subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Asegurar que los ejes sean iterables

    # Iterar sobre las columnas categóricas
    for i, col in enumerate(columns):
        sns.countplot(data=data, x=col, hue='target', ax=axes[i])
        axes[i].set_title(f'{col} vs. target')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # Eliminar gráficos vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar el layout
    plt.tight_layout()

    # Guardar y mostrar el gráfico
    plt.savefig('../assets/eda/categorical_target_countplot.png')
    plt.show()
    plt.close()

def plot_numerical_grid(data, columns, n_cols=3, figsize=(15, 10)):
    """
    Genera un grid de histogramas para variables numéricas.

    Args:
        data (DataFrame): Dataset que contiene los datos.
        columns (list): Lista de nombres de las columnas numéricas.
        n_cols (int): Número de columnas en el grid (por defecto, 3).
        figsize (tuple): Tamaño del gráfico (por defecto, (15, 10)).
    """
    # Calcular número de filas
    n_rows = (len(columns) + n_cols - 1) // n_cols

    # Crear subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Asegurar que los ejes sean iterables

    # Iterar sobre las columnas numéricas
    for i, col in enumerate(columns):
        sns.histplot(data=data, x=col, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribución de {col}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Conteo')

    # Eliminar gráficos vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Guardar y mostrar el gráfico
    plt.savefig('../assets/eda/numerical_histplot.png')
    plt.show()
    plt.close()

def plot_numerical_target_grid(data, columns, n_cols=3, figsize=(15, 10)):
    """
    Genera un grid de histogramas para variables numéricas.

    Args:
        data (DataFrame): Dataset que contiene los datos.
        columns (list): Lista de nombres de las columnas numéricas.
        n_cols (int): Número de columnas en el grid (por defecto, 3).
        figsize (tuple): Tamaño del gráfico (por defecto, (15, 10)).
    """
    # Calcular número de filas
    n_rows = (len(columns) + n_cols - 1) // n_cols

    # Crear subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Asegurar que los ejes sean iterables

    # Iterar sobre las columnas numéricas
    for i, col in enumerate(columns):
        sns.boxplot(data=data, x='target', y=col, ax=axes[i], hue='target')
        axes[i].set_title(f'{col} vs. target')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # Eliminar gráficos vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar el layout
    plt.tight_layout()

    # Guardar y mostrar el gráfico
    plt.savefig('../assets/eda/numerical_target_hist.png')
    plt.show()
    plt.close()

def plot_boxplot(data, columns, n_cols=3, figsize=(15, 10)):
    """
    Genera un grid de gráficos boxplot para
    identificar outliers.

    Args:
        data (DataFrame): Dataset que contiene los datos.
        columns (list): Lista de nombres de las columnas categóricas.
        n_cols (int): Número de columnas en el grid (por defecto, 3).
        figsize (tuple): Tamaño del gráfico (por defecto, (15, 10)).
    """
    # Calcular número de filas
    n_rows = (len(columns) + n_cols - 1) // n_cols

    # Crear subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Asegurar que los ejes sean iterables

    # Iterar sobre las columnas categóricas
    for i, col in enumerate(columns):
        sns.boxplot(data=data, y=col, hue='target', ax=axes[i])
        axes[i].set_title(f'Boxplot de {col}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # Eliminar gráficos vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar el layout
    plt.tight_layout()

    # Guardar y mostrar el gráfico
    plt.savefig('../assets/eda/boxplot_outliers.png')
    plt.show()
    plt.close()

def plot_corr_matrix(data, cmap="coolwarm", fmt=".2f"):
    # Matriz de correlación
    corr_matrix = data.corr()

    # Crear una máscara para la mitad superior
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Configurar el tamño del gráfico
    plt.figure(figsize=(10, 8))

    # Dibujar el heatmap con la máscara
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, fmt=fmt, linewidth=.5)
    plt.title('Matriz de correlación')

    # Guardar el gráfico
    plt.savefig('../assets/eda/correlation_matrix.png')

    # Mostrar el gráfico
    plt.show()

    # Cerrar para liberar memoria
    plt.close()

# ------------ Modelado ------------
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Ajusta y evalúa los modelos de aprendizaje automático dados.
    modelos : un diccionario de diferentes modelos de aprendizaje automático de Scikit-Learn
    X_entrenamiento : datos de entrenamiento (sin etiquetas)
    X_prueba : datos de prueba (sin etiquetas)
    y_entrenamiento : etiquetas de entrenamiento
    y_prueba : etiquetas de prueba
    """
    # Establecer semilla aleatoria
    np.random.seed(42)
    # Hacer un diccionario para mantener las puntuaciones de los modelos
    model_scores = {}
    # Iterar a través de los modelos
    for name, model in models.items():
        # Ajustar el modelo a los datos
        model.fit(X_train, y_train)
        # Evaluar el modelo y agregar su puntuación a model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo usando métricas avanzadas.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Classification report
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC
    if y_pred_proba is not None:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.2f}")

def plot_conf_mat(y_test, y_preds, model='', path_plot='../assets/modeling/confusion_matrix.png'):
    """
    Grafica una matriz de confusión usando heatmap() de Seaborn.
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    cm = confusion_matrix(y_test, y_preds)
    ax = sns.heatmap(cm,
                     annot=True, # Anotar los recuadros
                     cbar=False)
    plt.title(f"Matriz de correlación {model}", fontsize=12)
    plt.xlabel("Etiqueta predicha", fontsize=10)
    plt.ylabel("Etiqueta verdadera", fontsize=10)

    # Guardar el gráfico
    plt.savefig(path_plot)

    # Mostrar el gráfico
    plt.show()

    # Cerrar para liberar memoria
    plt.close()

def plot_roc_curve(model, X_test, y_test):
    """
    Grafica la curva ROC para un modelo dado.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Curva ROC")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_cross_validation_metrics(models,
                                      X_train,
                                      y_train,
                                      cv=5,
                                      metrics=None,
                                      scaled_models=None,
                                      X_train_scaled=None):
    """
    Evalúa múltiples métricas mediante validación cruzada para diferentes modelos.

    Args:
        models (dict): Diccionario con los modelos a evaluar (nombre: instancia).
        X_train (array): Datos de entrenamiento.
        y_train (array): Etiquetas de entrenamiento.
        cv (int): Número de folds para la validación cruzada.
        metrics (list): Lista de métricas a evaluar (por defecto: ["accuracy", "precision", "recall", "f1"]).
        scaled_models (list): Lista de nombres de modelos que necesitan datos escalados (opcional).

    Returns:
        dict: Diccionario con las métricas promedio y desviaciones estándar por modelo y métrica.
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]
    
    results = {}
    
    for model_name, model in models.items():
        results[model_name] = {}
        for metric in metrics:
            # Verificar si el modelo necesita datos escalados
            if scaled_models and model_name in scaled_models:
                scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring=metric)
            else:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
            
            # Guardar métricas promedio y desviación estándar
            results[model_name][metric] = {
                "mean": scores.mean(),
                "std": scores.std()
            }
            print(f"{model_name} - {metric.capitalize()}: {scores.mean():.2f} (+/- {scores.std():.2f})")
    
    return results

def plot_cross_validation_metrics(metrics_results,
                                  path_plot="../assets/modeling/plot_cross_validation_metrics.png"):
    """
    Genera una gráfica de barras agrupadas para comparar las métricas de validación cruzada entre modelos.

    Args:
        metrics_results (dict): Resultados de la validación cruzada por modelo y métrica.
            Formato: { "Modelo": {"Métrica1": {"mean": valor, "std": valor}, ...}, ...}

    Returns:
        None
    """
    # Crear un DataFrame a partir de los resultados
    data = {
        model: {metric: metrics_results[model][metric]["mean"] for metric in metrics_results[model]}
        for model in metrics_results
    }
    df = pd.DataFrame(data)

    # Configurar gráfico
    ax = df.plot(kind="bar", figsize=(8, 6), width=0.8)

    # Personalizar el gráfico
    plt.title("Métricas de Validación Cruzada por modelo", fontsize=14)
    plt.xlabel("Métricas", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(rotation=0, fontsize=10)
    plt.ylim(0, 1)  # Asegurar que las métricas estén entre 0 y 1
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Mostrar el gráfico
    plt.tight_layout()

    # Guardar el gráfico
    plt.savefig(path_plot)

    # Mostrar el gráfico
    plt.show()

    # Cerrar para liberar memoria
    plt.close()

