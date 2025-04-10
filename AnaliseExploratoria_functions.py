import pandas as pd

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