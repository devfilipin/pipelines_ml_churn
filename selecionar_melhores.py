import pandas as pd

from config import PASTA_METRICS, criar_pastas_resultados


ARQUIVO_BASELINES = PASTA_METRICS / "baseline_results.csv"
ARQUIVO_SELECIONADOS = PASTA_METRICS / "selected_models.csv"
COLUNAS_ORDENACAO = ["f1_churn_mean", "recall_churn_mean", "pr_auc_mean"]
COLUNAS_SAIDA = [
    "model",
    "balancing_scenario",
    "f1_churn_mean",
    "recall_churn_mean",
    "pr_auc_mean",
    "precision_churn_mean",
    "roc_auc_mean",
    "accuracy_mean",
]


def carregar_resultados_baseline() -> pd.DataFrame:
    # Carrega os resultados gerados pelo script de treinamento dos baselines.
    if not ARQUIVO_BASELINES.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_BASELINES}. "
            "Execute treinar_baselines.py antes de selecionar os modelos."
        )

    return pd.read_csv(ARQUIVO_BASELINES)


def preparar_colunas(df_resultados: pd.DataFrame) -> pd.DataFrame:
    # Padroniza os nomes das colunas para o arquivo final de seleção.
    df_resultados = df_resultados.rename(
        columns={
            "modelo": "model",
            "cenario_balanceamento": "balancing_scenario",
        }
    )

    colunas_obrigatorias = [*COLUNAS_SAIDA, *COLUNAS_ORDENACAO]
    colunas_faltantes = [
        coluna for coluna in colunas_obrigatorias if coluna not in df_resultados.columns
    ]

    if colunas_faltantes:
        raise ValueError(
            "O arquivo de resultados não possui as colunas esperadas: "
            f"{colunas_faltantes}"
        )

    return df_resultados


def selecionar_melhores(df_resultados: pd.DataFrame, quantidade: int = 3) -> pd.DataFrame:
    # Ordena pelos critérios principais e mantém apenas os melhores candidatos.
    return (
        df_resultados.sort_values(
            by=COLUNAS_ORDENACAO,
            ascending=False,
        )
        .head(quantidade)
        .loc[:, COLUNAS_SAIDA]
    )


def main() -> None:
    criar_pastas_resultados()

    df_resultados = carregar_resultados_baseline()
    df_resultados = preparar_colunas(df_resultados)
    df_selecionados = selecionar_melhores(df_resultados)

    df_selecionados.to_csv(ARQUIVO_SELECIONADOS, index=False)

    print("Modelos selecionados para otimização:")
    print(df_selecionados.to_string(index=False))
    print(f"\nArquivo salvo em: {ARQUIVO_SELECIONADOS}")


if __name__ == "__main__":
    main()
