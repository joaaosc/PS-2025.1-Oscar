import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy.stats import probplot
from sklearn.impute import SimpleImputer, KNNImputer

def calcular_resultado(df):
    """
    Calcula o resultado do confronto com base nas colunas 'Gols 1' e 'Gols 2'.
    - O DataFrame original não é modificado.
    - O resultado é retornado em uma pandas Series com o nome "resultados".

    Parâmetros:
    df (pandas.DataFrame): DataFrame que contém as colunas 'Gols 1' e 'Gols 2'.

    Retorna:
    pandas.Series: Série com os resultados, onde:
        - "Vitória" se 'Gols 1' > 'Gols 2'
        - "Empate" se 'Gols 1' == 'Gols 2'
        - "Derrota" se 'Gols 1' < 'Gols 2'
    """
    resultados = []
    for _, row in df.iterrows():
        if row['Gols 1'] > row['Gols 2']:
            resultados.append("Vitória")
        elif row['Gols 1'] == row['Gols 2']:
            resultados.append("Empate")
        else:
            resultados.append("Derrota")
    return pd.Series(resultados, name="resultado")

def substituir_tiros_por_chutes(df):
    """
    Recebe um DataFrame e retorna uma nova cópia onde:
      - Em 'Tiro de meta 2', se houver NaN, o valor é substituído pelo de 'Chutes fora 1'.
      - Em 'Tiro de meta 1', se houver NaN, o valor é substituído pelo de 'Chutes fora 2'.

    O DataFrame original não é modificado.
    """

    df_copy = df.copy()

    # Criação de máscara para linhas onde 'Tiros de Meta 2' é NaN
    mask_tiros_meta2 = df_copy['Tiro de meta 2'].isnull()
    # Substitui pelos valores correspondentes da coluna 'Chute Fora 1'
    df_copy.loc[mask_tiros_meta2, 'Tiro de meta 2'] = df_copy.loc[mask_tiros_meta2, 'Chutes fora 1']

    # Análogo para 'Tiros de Meta 1'
    mask_tiros_meta1 = df_copy['Tiro de meta 1'].isnull()
    df_copy.loc[mask_tiros_meta1, 'Tiro de meta 1'] = df_copy.loc[mask_tiros_meta1, 'Chutes fora 2']

    return df_copy

from typing import Tuple
def histogram_and_stats(df: pd.DataFrame, column: str, plot: bool = True, interactive: bool = True, bins: int = 40)\
        -> Tuple[int, int, float, float, float, float, float, float, float, float, float]:
    """
    Plota (opcionalmente) um histograma para a coluna especificada de um DataFrame e retorna e printa estatísticas descritivas.

    :param df: O DataFrame que contém os dados.
    :type df: pd.DataFrame
    :param column: O nome da coluna para a qual será gerado o histograma.
    :type column: str
    :param plot: Se True, gera o gráfico. Se False, apenas retorna as estatísticas.
    :type plot: bool
    :param interactive: Se True, o plot é feito pelo plotly (interativo. Se False, pelo matplotib.
    :type interactive: bool
    :param bins: Ajuste para o gráfico do histograma alinhar corretamente.
    :type bins: int


    :return: Uma tupla contendo:
        - num_nan (int): Número de valores NaN
        - num_valid (int): Número de valores não NaN
        - media (float): Média dos valores (excluindo NaN)
        - desvio (float): Desvio padrão dos valores (excluindo NaN)
        - minimo (float): Valor mínimo (excluindo NaN)
        - maximo (float): Valor máximo (excluindo NaN)
        - percentil_5 (float): 5º percentil
        - percentil_25 (float): 25º percentil (1º quartil)
        - percentil_50 (float): 50º percentil (mediana)
        - percentil_75 (float): 75º percentil (3º quartil)
        - percentil_95 (float): 95º percentil
    :rtype: Tuple[int, int, float, float, float, float, float, float, float, float, float]

    Exemplo de uso:
        >>> stats = histogram_and_stats(df, 'Chutes a gol 1')
        >>> print("Estatísticas:", stats)
    """

    # Seleciona a coluna e remove os NaN para o histograma
    data = df[column]
    data_sem_nan = data.dropna()

    # Cálculo das estatísticas
    num_nan = data.isna().sum()
    num_valid = data.notna().sum()
    media = data_sem_nan.mean()
    desvio = data_sem_nan.std()
    minimo = data_sem_nan.min()
    maximo = data_sem_nan.max()
    percentil_5 = data_sem_nan.quantile(0.05)
    percentil_25 = data_sem_nan.quantile(0.25)
    percentil_50 = data_sem_nan.quantile(0.50)
    percentil_75 = data_sem_nan.quantile(0.75)
    percentil_95 = data_sem_nan.quantile(0.95)

    if plot == True:
        if not interactive:
            plt.figure(figsize=(6, 4))
            plt.hist(data_sem_nan, bins=bins, edgecolor='black', histtype='stepfilled')
            plt.title(f'{column}')
            plt.xlabel(column)
            plt.ylabel('Ocorrências')
            plt.grid(False)
            plt.axvline(media, linewidth=1, label='Média')
            plt.axvline(percentil_50, linewidth=1, label='Mediana')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            fig = px.histogram(
                x=data_sem_nan,
                #nbins='auto',
                title=f'{column}',
                labels={"x": column, "y": "Ocorrências"}
            )

            # Adiciona linhas verticais para a média e a mediana
            fig.add_vline(
                x=media,
                line_width=1,
                line_dash="dash",
                line_color="red",
                annotation_text="Média",
                annotation_position="top right"
            )
            fig.add_vline(
                x=percentil_50,
                line_width=1,
                line_dash="dash",
                line_color="green",
                annotation_text="Mediana",
                annotation_position="top right"
            )
            fig.show()

    print(
        f"\033[1mNaN's:\033[0m {num_nan} --|-- \033[1mOcorrências:\033[0m {num_valid} --|-- \033[1mMédia:\033[0m {media:.2f} --|-- "
        f"\033[1mDesv. padrão:\033[0m {desvio:.2f} --|-- \033[1mMínimo:\033[0m {minimo:.2f} --|-- \033[1mMáximo:\033[0m {maximo:.2f} --|-- "
        f"\033[1m5º percentil:\033[0m {percentil_5:.2f} --|-- \033[1m25º percentil:\033[0m {percentil_25:.2f} --|-- "
        f"\033[1m50º percentil (mediana):\033[0m {percentil_50:.2f} --|-- \033[1m75º percentil:\033[0m {percentil_75:.2f} --|-- "
        f"\033[1m95º percentil:\033[0m {percentil_95:.2f}\n--------------")

    return (int(num_nan),int(num_valid),float(media),float(desvio),int(minimo),int(maximo),float(percentil_5),float(percentil_25),float(percentil_50),float(percentil_75),float(percentil_95))


def evaluate_distribution(df: pd.DataFrame, column: str, show_boxplot: bool = False,bins: int = 40) -> dict:
    """
    Avalia a distribuição de uma coluna do DataFrame calculando skewness e
    (excesso de) kurtosis e gerando gráficos para visualização: histograma com KDE,
    Q-Q plot e opcionalmente um boxplot.

    Parameters:
        df (pd.DataFrame): DataFrame contendo os dados.
        column (str): Nome da coluna a ser avaliada.
        show_boxplot (bool, optional): Se True, inclui o boxplot na visualização.
                                       Padrão é False.
        bins (int, optional): Número de intervalos no histograma. O default é 40, mas pode ser alterado para
                    uma melhor visualização

    Returns:
        dict: Dicionário com os seguintes valores:
            - "skewness": Valor da assimetria da distribuição.
            - "kurtosis": Valor do excesso de kurtosis (kurtosis - 3).
    """
    # Remove os NaN para os cálculos e plots
    dados = df[column].dropna()

    skewness_value = dados.skew()
    kurtosis_value = dados.kurt()

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(dados, kde=True,bins=bins)
    plt.title(f'Histograma de {column}\nSkew: {skewness_value:.2f} | Kurtosis (excesso): {kurtosis_value:.2f}')

    # Q-Q Plot
    plt.subplot(1, 3, 2)
    probplot(dados, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot de {column}')

    plt.subplot(1, 3, 3)
    if show_boxplot:
        sns.boxplot(x=dados)
        plt.title(f'Boxplot de {column}')
    else:
        plt.axis('off')

    plt.show()

    print(f"\nColuna: {column}")
    print(f"Skewness: {skewness_value:.3f}")
    print(f"Kurtosis (excesso): {kurtosis_value:.3f}")
    return {
        "skewness": skewness_value,
        "kurtosis": kurtosis_value
    }

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

def impute_missing_values(df: pd.DataFrame, columns: list or str, strategy: str = "median",
                          n_neighbors: int = 5) -> pd.DataFrame:
    """
    Imputa os valores faltantes (NaN) nas colunas especificadas do DataFrame utilizando
    a estratégia definida. As opções de estratégia são:

    - "mean": substitui os NaN pela média dos valores.
    - "median": substitui os NaN pela mediana dos valores.
    - "knn": utiliza o KNN Imputer para imputar os NaN considerando os vizinhos mais próximos.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados.
        columns (list or str): Lista de nomes das colunas onde os valores faltantes serão imputados,
                               ou uma string para uma única coluna.
        strategy (str, opcional): Estratégia de imputação ("mean", "median" ou "knn").
                                  Padrão é "median".
        n_neighbors (int, opcional): Número de vizinhos a serem considerados no KNN Imputer
                                     (apenas se strategy for "knn"). Padrão é 5.

    Retorna:
        pd.DataFrame: DataFrame atualizado com os valores faltantes imputados.

    Raises:
        ValueError: Se a estratégia fornecida não for "mean", "median" ou "knn".
    """
    if strategy not in ["mean", "median", "knn"]:
        raise ValueError("A estratégia deve ser 'mean', 'median' ou 'knn'.")

    # Se 'columns' for uma string, converte para lista
    if isinstance(columns, str):
        columns = [columns]

    # Usamos .loc para evitar o SettingWithCopyWarning
    if strategy in ["mean", "median"]:
        imputer = SimpleImputer(strategy=strategy)
        df.loc[:, columns] = imputer.fit_transform(df.loc[:, columns])
    elif strategy == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df.loc[:, columns] = imputer.fit_transform(df.loc[:, columns])

    return df

# Exemplo de uso:
# df = impute_missing_values(df, 'Posse 1(%)', strategy='mean')
# df = impute_missing_values(df, 'Posse 2(%)', strategy='mean')
