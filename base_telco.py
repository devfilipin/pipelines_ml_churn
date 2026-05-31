import requests
from pathlib import Path
import pandas as pd


URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

PASTA_DESTINO = Path("data")
ARQUIVO_DESTINO = PASTA_DESTINO / "Telco-Customer-Churn.csv"


def baixar_telco_churn():
    PASTA_DESTINO.mkdir(parents=True, exist_ok=True)

    response = requests.get(URL, timeout=30)
    response.raise_for_status()

    ARQUIVO_DESTINO.write_bytes(response.content)

    df = pd.read_csv(ARQUIVO_DESTINO)

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


if __name__ == "__main__":
    df = baixar_telco_churn()