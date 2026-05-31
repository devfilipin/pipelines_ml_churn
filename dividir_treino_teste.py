from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from base_telco import ARQUIVO_DESTINO, baixar_telco_churn


# Coluna que queremos prever nos modelos de IA.
COLUNA_ALVO = "Churn"

# Conversão da variável alvo: No vira 0 e Yes vira 1.
MAPA_CHURN = {"No": 0, "Yes": 1}

# customerID é apenas um identificador do cliente, não uma característica útil
# para treinar o modelo.
COLUNAS_REMOVER = ["customerID"]

# 20% da base será reservado para teste e 80% ficará para treinamento.
TAMANHO_TESTE = 0.2

# A semente garante que a divisão seja sempre igual a cada execução.
SEMENTE_ALEATORIA = 42

# Pasta onde os arquivos de treino e teste serão salvos.
PASTA_SAIDA = Path("data") / "treino_teste"


def carregar_base() -> pd.DataFrame:
    # Se o CSV já existe localmente, apenas lemos o arquivo.
    if ARQUIVO_DESTINO.exists():
        return pd.read_csv(ARQUIVO_DESTINO)

    # Caso contrário, usamos a função do base_telco.py para baixar a base.
    return baixar_telco_churn()


def separar_treino_teste(
    df: pd.DataFrame,
    coluna_alvo: str = COLUNA_ALVO,
    tamanho_teste: float = TAMANHO_TESTE,
    semente_aleatoria: int = SEMENTE_ALEATORIA,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Garante que a coluna que queremos prever realmente existe na base.
    if coluna_alvo not in df.columns:
        raise ValueError(f"A coluna alvo '{coluna_alvo}' não existe na base.")

    # Converte a coluna alvo para valores numéricos antes de treinar o modelo.
    df = df.copy()
    df[coluna_alvo] = df[coluna_alvo].map(MAPA_CHURN)

    if df[coluna_alvo].isna().any():
        raise ValueError(
            f"A coluna '{coluna_alvo}' possui valores diferentes de Yes/No."
        )

    # Remove somente as colunas configuradas que existirem no DataFrame.
    colunas_remover = [coluna for coluna in COLUNAS_REMOVER if coluna in df.columns]

    # X contém as variáveis de entrada; y contém a resposta que o modelo aprende.
    x = df.drop(columns=[coluna_alvo, *colunas_remover])
    y = df[coluna_alvo].astype(int)

    # stratify=y mantém a proporção de 0 e 1 parecida em treino e teste.
    return train_test_split(
        x,
        y,
        test_size=tamanho_teste,
        random_state=semente_aleatoria,
        stratify=y,
    )


def salvar_conjuntos(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    pasta_saida: Path = PASTA_SAIDA,
) -> None:
    # Cria a pasta de saída caso ela ainda não exista.
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # Salva entradas (X) e respostas (y) separadamente para facilitar o uso
    # posterior em modelos de machine learning.
    x_train.to_csv(pasta_saida / "X_train.csv", index=False)
    x_test.to_csv(pasta_saida / "X_test.csv", index=False)
    y_train.to_csv(pasta_saida / "y_train.csv", index=False)
    y_test.to_csv(pasta_saida / "y_test.csv", index=False)


def main() -> None:
    # Fluxo principal: carregar, dividir, salvar e mostrar um resumo.
    df = carregar_base()
    x_train, x_test, y_train, y_test = separar_treino_teste(df)
    salvar_conjuntos(x_train, x_test, y_train, y_test)

    print("Conjuntos gerados com sucesso.")
    print(f"Treinamento: {x_train.shape[0]} linhas e {x_train.shape[1]} colunas")
    print(f"Teste: {x_test.shape[0]} linhas e {x_test.shape[1]} colunas")
    print(f"Arquivos salvos em: {PASTA_SAIDA}")

    # Mostra se a proporção da variável alvo ficou equilibrada entre os conjuntos.
    print("\nDistribuição de Churn no treino:")
    print(y_train.value_counts(normalize=True))

    print("\nDistribuição de Churn no teste:")
    print(y_test.value_counts(normalize=True))


if __name__ == "__main__":
    main()
