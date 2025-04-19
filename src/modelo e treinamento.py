
from typing import Tuple, Union, Dict
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np

# ========================
#   Funções de modelos
# ========================

def get_xgboost_model(X_train: Union[pd.DataFrame, np.ndarray], 
                    y_train: Union[pd.Series, np.ndarray]
                    ) -> XGBClassifier:
    """
    Faz tuning e treina um XGBoost com RandomizedSearchCV.

    Parâmetros:
    - X_train (DataFrame ou ndarray): Dados de treino.
    - y_train (Series ou ndarray): Rótulos.

    Retorna:
    - XGBClassifier treinado com os melhores hiperparâmetros.
    """


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



def get_mlp_model(X_train: Union[pd.DataFrame, np.ndarray], 
                y_train: Union[pd.Series, np.ndarray], 
                hidden_layers: Tuple[int, ...] = (128, 64, 32),
                alpha: float = 1e-4,
                max_iter: int = 200
                ) -> Pipeline:
    """
    Treina uma MLP com StandardScaler dentro de um pipeline.

    Parâmetros:
    - X_train (DataFrame ou ndarray): Dados de treino.
    - y_train (Series ou ndarray): Rótulos.
    - hidden_layers (Tuple[int, ...]): Arquitetura da rede.
    - alpha (float): Termo de regularização L2.
    - max_iter (int): Número máximo de épocas.

    Retorna:
    - Pipeline sklearn com scaler e MLP treinada.
    """


    mlp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate='adaptive',
            max_iter=max_iter,
            random_state=42,
            verbose=True
        ))
    ])
    mlp_pipeline.fit(X_train, y_train)

    print("\n[LOG - MLP] Arquitetura usada:")
    print(f"  hidden_layer_sizes: {hidden_layers}")
    print(f"  max_iter: {max_iter}")

    return mlp_pipeline


def build_ensemble(xgb_model: XGBClassifier, mlp_model: Pipeline) -> VotingClassifier:
    """
    Cria um ensemble soft voting entre XGBoost e MLP.

    Parâmetros:
    - xgb_model (XGBClassifier): Modelo XGBoost treinado.
    - mlp_model (Pipeline): Pipeline com MLP treinada.

    Retorna:
    - VotingClassifier com soft voting.
    """

    ensemble = VotingClassifier(estimators=[
        ('xgb', xgb_model),
        ('mlp', mlp_model)
    ], voting='soft')
    return ensemble



# ===========================
#   Funções de treinamento
# ===========================

def split(file_path: str, target_column: str = "target", test_size: float = 0.2, random_state: int = 42
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Lê o CSV, separa features/target e realiza o split em treino e teste.

    Parâmetros:
    - file_path (str): Caminho para o arquivo CSV.
    - target_column (str): Nome da coluna alvo.
    - test_size (float): Proporção dos dados para teste.
    - random_state (int): Semente aleatória.

    Retorna:
    - Tuple contendo X_train, X_test, y_train, y_test
    """


    df = pd.read_csv(file_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



def ensemble_train(X_train: Union[pd.DataFrame, np.ndarray],
                y_train: Union[pd.Series, np.ndarray]
                ) -> VotingClassifier:
    """
    Treina os modelos base e o ensemble.

    Retorna:
    - VotingClassifier treinado.
    """
    print("Treinando XGBoost...")
    xgb_model = get_xgboost_model(X_train, y_train)

    print("Treinando MLP...")
    mlp_model = get_mlp_model(X_train, y_train)

    print("Treinando ensemble...")
    ensemble = build_ensemble(xgb_model, mlp_model)
    ensemble.fit(X_train, y_train)

    return ensemble




def ensemble_test(ensemble_model: VotingClassifier,
                X_test: Union[pd.DataFrame, np.ndarray]
                ) -> np.ndarray:
    """
    Realiza a predição do ensemble.

    Retorna:
    - y_pred (np.ndarray): Predições.
    """
    return ensemble_model.predict(X_test)


def ensemble_evaluate(y_test: Union[pd.Series, np.ndarray],
                    y_pred: Union[pd.Series, np.ndarray]
                    ) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """
    Avalia o desempenho do ensemble.

    Retorna:
    - dicionário com accuracy, classification report e matriz de confusão.
    """
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    print("\n==== Avaliação ====")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix (normalizada):")
    print(cm)

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


if __name__ == "__main__":
# Import a ser utilizado nos notebooks:
#    from model_utils import (
#    split,
#   ensemble_train,
#    ensemble_test,
#    ensemble_evaluate
#)



    
    file_path = "caminho para o dataset.csv"  # Altere para o caminho real do seu dataset

    
    print("Splitando dados...")
    X_train, X_test, y_train, y_test = split(file_path, target_column="target")

    # === Treinamento do Ensemble ===
    print("\nTreinamento o ensemble...")
    ensemble_model = ensemble_train(X_train, y_train)

    # === Teste ===
    print("\nTestando o ensemble...")
    y_pred = ensemble_test(ensemble_model, X_test)

    # === Avaliação ===
    print("\nAvaliação do modelo:")
    metrics = ensemble_evaluate(y_test, y_pred)

