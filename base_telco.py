from pathlib import Path
import pandas as pd


PASTA_DESTINO = Path("data")
ARQUIVO_CUSTOMER_CHURN = PASTA_DESTINO / "CustomerChurn.xlsx"
ARQUIVO_DEMOGRAFICO = PASTA_DESTINO / "Telco_customer_churn_demographics.xlsx"
ARQUIVO_DESTINO = PASTA_DESTINO / "Telco-Customer-Churn.csv"
COLUNA_CHAVE = "Customer ID"
COLUNAS_DEMOGRAFICAS = [COLUNA_CHAVE, "Gender", "Age"]


def consolidar_telco_churn() -> pd.DataFrame:
    PASTA_DESTINO.mkdir(parents=True, exist_ok=True)

    df_churn = pd.read_excel(ARQUIVO_CUSTOMER_CHURN)
    df_demografico = pd.read_excel(ARQUIVO_DEMOGRAFICO, usecols=COLUNAS_DEMOGRAFICAS)

    df = df_churn.merge(
        df_demografico,
        on=COLUNA_CHAVE,
        how="left",
        validate="one_to_one",
    )

    if df[["Gender", "Age"]].isna().any(axis=1).any():
        raise ValueError("Existem clientes sem Gender ou Age após a consolidação.")

    df.to_csv(ARQUIVO_DESTINO, index=False)

    print("\nPrévia da base:")
    print(df.head())

    print("\nInformações gerais:")
    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")
    print(f"Colunas disponíveis: {list(df.columns)}")

    if "Churn" in df.columns:
        print("\nDistribuição da variável Churn:")
        print(df["Churn"].value_counts())

    return df


def baixar_telco_churn() -> pd.DataFrame:
    return consolidar_telco_churn()


if __name__ == "__main__":
    df = consolidar_telco_churn()
