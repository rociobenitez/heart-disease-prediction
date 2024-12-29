import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def train_random_forest(X_train, y_train, param_grid=None):
    """
    Entrena un modelo Random Forest con búsqueda de hiperparámetros.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": np.arange(10, 1000, 50),
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": np.arange(2, 20, 2),
            "min_samples_leaf": np.arange(1, 20, 2)
        }
    grid = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                              param_distributions=param_grid,
                              cv=5,
                              n_iter=20,
                              scoring='accuracy',
                              random_state=42)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def train_logistic_regression(X_train, y_train, param_grid=None):
    """
    Entrena un modelo Logistic Regression con búsqueda de hiperparámetros.
    """
    if param_grid is None:
        param_grid = {
            "C": np.logspace(-4, 4, 20),
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        }
    grid = GridSearchCV(LogisticRegression(),
                        param_grid=param_grid,
                        cv=5,
                        verbose=True)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def train_knn(X_train, y_train, param_grid=None):
    """
    Entrena un modelo KNN con búsqueda de hiperparámetros.
    """
    if param_grid is None:
        param_grid = {
            "n_neighbors": np.arange(1, 21, 1),
            "weights": ["uniform", "distance"]
        }
    grid = GridSearchCV(KNeighborsClassifier(),
                        param_grid=param_grid,
                        cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
