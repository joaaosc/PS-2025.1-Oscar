import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

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
def plot_histogram_and_stats(df: pd.DataFrame, column: str, interactive=True) -> Tuple[int, int, float, float, float, float, float, float, float, float, float]:
    """
    Plota um histograma para a coluna especificada de um DataFrame e retorna e printa estatísticas descritivas.

    :param df: O DataFrame que contém os dados.
    :type df: pd.DataFrame
    :param column: O nome da coluna para a qual será gerado o histograma.
    :type column: str

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
        >>> stats = plot_histogram_and_stats(df, 'Chutes a gol 1')
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

    # Plot do histograma
    if interactive == False:
        plt.figure(figsize=(6, 4))
        plt.hist(data_sem_nan, bins='auto', edgecolor='black', histtype='stepfilled')
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
        f"NaN's: {num_nan} | Ocorrências: {num_valid} | Média: {media:.2f} | "
        f"Desv. padrão: {desvio:.2f} | Mínimo: {minimo:.2f} | Máximo: {maximo:.2f} | "
        f"5º percentil: {percentil_5:.2f} | 25º percentil: {percentil_25:.2f} | "
        f"50º percentil (mediana): {percentil_50:.2f} | 75º percentil: {percentil_75:.2f} | "
        f"95º percentil: {percentil_95:.2f}")

# Retorna as estatísticas em uma tupla
    return (int(num_nan),int(num_valid),float(media),float(desvio),int(minimo),int(maximo),float(percentil_5),float(percentil_25),float(percentil_50),float(percentil_75),float(percentil_95))

