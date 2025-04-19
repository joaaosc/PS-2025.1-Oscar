

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ========================
#   Funções de modelos
# ========================

def get_xgboost_model(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    rand_search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42
    )
    rand_search.fit(X_train, y_train)

    print("\n[LOG - XGBoost] Melhores hiperparâmetros encontrados:")
    for param, value in rand_search.best_params_.items():
        print(f"  {param}: {value}")

    return rand_search.best_estimator_


def get_mlp_model(X_train, y_train):
    hidden_layers = (64, 32)
    max_iter = 500

    mlp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=hidden_layers,
                            max_iter=max_iter,
                            random_state=42))
    ])
    mlp_pipeline.fit(X_train, y_train)

    print("\n[LOG - MLP] Arquitetura usada:")
    print(f"  hidden_layer_sizes: {hidden_layers}")
    print(f"  max_iter: {max_iter}")

    return mlp_pipeline


def build_ensemble(xgb_model, mlp_model):
    ensemble = VotingClassifier(estimators=[
        ('xgb', xgb_model),
        ('mlp', mlp_model)
    ], voting='soft')
    return ensemble

def split(file_path):
    df = pd.read_csv("file_path") 
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def ensemble_train(X_train, y_train):
    # === Modelos individuais ===
    print("Treinando XGBoost...")
    xgb_model = get_xgboost_model(X_train, y_train)

    print("Treinando MLP...")
    mlp_model = get_mlp_model(X_train, y_train)

    # === Ensemble ===
    print("Treinando ensemble...")
    ensemble = build_ensemble(xgb_model, mlp_model)
    ensemble.fit(X_train, y_train)

    return ensemble

def ensemble_test(ensemble_model, X_test):
    y_pred = ensemble_model.predict(X_test)

    return y_pred

def ensemble_evaluate(y_test, y_pred):
    print("\n==== Avaliação ====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred, normalize='true'))


if __name__ == "__main__":

    pass