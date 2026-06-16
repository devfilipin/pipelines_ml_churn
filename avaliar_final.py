import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import PASTA_FIGURES, PASTA_METRICS, criar_pastas_resultados
from preprocessamento import carregar_conjuntos


ARQUIVO_RESULTADOS_OTIMIZADOS = PASTA_METRICS / "optimized_results.csv"
ARQUIVO_METRICAS_FINAIS = PASTA_METRICS / "final_test_metrics.csv"
ARQUIVO_PREDICOES_FINAIS = PASTA_METRICS / "final_predictions.csv"
FIGURA_MATRIZ_CONFUSAO = PASTA_FIGURES / "confusion_matrix_best_model.png"
FIGURA_ROC = PASTA_FIGURES / "roc_curve_best_model.png"
FIGURA_PR = PASTA_FIGURES / "pr_curve_best_model.png"
THRESHOLD_PADRAO = 0.5


def carregar_melhor_modelo_otimizado() -> tuple[pd.Series, object]:
    # Seleciona o melhor modelo otimizado pela maior pontuação de validação.
    if not ARQUIVO_RESULTADOS_OTIMIZADOS.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_RESULTADOS_OTIMIZADOS}. "
            "Execute otimizar_modelos.py antes da avaliação final."
        )

    df_resultados = pd.read_csv(ARQUIVO_RESULTADOS_OTIMIZADOS)

    if df_resultados.empty:
        raise ValueError("O arquivo de resultados otimizados está vazio.")

    colunas_necessarias = ["model", "balancing_scenario", "best_score", "model_path"]
    colunas_faltantes = [
        coluna for coluna in colunas_necessarias if coluna not in df_resultados.columns
    ]

    if colunas_faltantes:
        raise ValueError(
            "O arquivo de resultados otimizados não possui as colunas esperadas: "
            f"{colunas_faltantes}"
        )

    melhor_linha = df_resultados.sort_values("best_score", ascending=False).iloc[0]
    caminho_modelo = melhor_linha["model_path"]
    modelo = joblib.load(caminho_modelo)

    return melhor_linha, modelo


def obter_score_churn(modelo: object, X_test: pd.DataFrame) -> pd.Series | None:
    # Obtém probabilidade da classe churn ou score contínuo quando possível.
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X_test)
        return pd.Series(proba[:, 1], index=X_test.index, name="y_proba_churn")

    if hasattr(modelo, "decision_function"):
        scores = modelo.decision_function(X_test)
        return pd.Series(scores, index=X_test.index, name="y_proba_churn")

    return None


def gerar_predicoes(
    modelo: object,
    X_test: pd.DataFrame,
) -> tuple[pd.Series, pd.Series | None]:
    # Usa threshold 0.5 quando houver probabilidade; caso contrário, usa predict.
    score_churn = obter_score_churn(modelo, X_test)

    if score_churn is not None and hasattr(modelo, "predict_proba"):
        y_pred = (score_churn >= THRESHOLD_PADRAO).astype(int)
    else:
        y_pred = pd.Series(modelo.predict(X_test), index=X_test.index, name="y_pred")

    return y_pred, score_churn


def calcular_metricas(
    y_test: pd.Series,
    y_pred: pd.Series,
    score_churn: pd.Series | None,
) -> dict[str, float | None]:
    # Calcula métricas finais no conjunto de teste independente.
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_churn": precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "recall_churn": recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "f1_churn": f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "roc_auc": None,
        "pr_auc": None,
    }

    if score_churn is not None:
        metricas["roc_auc"] = roc_auc_score(y_test, score_churn)
        metricas["pr_auc"] = average_precision_score(y_test, score_churn)

    return metricas


def salvar_predicoes(
    y_test: pd.Series,
    y_pred: pd.Series,
    score_churn: pd.Series | None,
) -> None:
    # Salva as predições finais para análise posterior.
    df_predicoes = pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "y_pred": pd.Series(y_pred).reset_index(drop=True),
        }
    )

    if score_churn is not None:
        df_predicoes["y_proba_churn"] = score_churn.reset_index(drop=True)

    df_predicoes.to_csv(ARQUIVO_PREDICOES_FINAIS, index=False)


def salvar_matriz_confusao(y_test: pd.Series, y_pred: pd.Series) -> None:
    # Salva a matriz de confusão do melhor modelo no teste.
    matriz = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["No Churn", "Churn"],
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de Confusão - Melhor Modelo")
    plt.tight_layout()
    plt.savefig(FIGURA_MATRIZ_CONFUSAO, dpi=150)
    plt.close()


def salvar_curva_roc(y_test: pd.Series, score_churn: pd.Series | None) -> None:
    # Salva a curva ROC quando há score contínuo disponível.
    if score_churn is None:
        return

    fpr, tpr, _ = roc_curve(y_test, score_churn)
    auc = roc_auc_score(y_test, score_churn)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.title("Curva ROC - Melhor Modelo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURA_ROC, dpi=150)
    plt.close()


def salvar_curva_precision_recall(
    y_test: pd.Series,
    score_churn: pd.Series | None,
) -> None:
    # Salva a curva Precision-Recall quando há score contínuo disponível.
    if score_churn is None:
        return

    precision, recall, _ = precision_recall_curve(y_test, score_churn)
    pr_auc = average_precision_score(y_test, score_churn)

    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall - Melhor Modelo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURA_PR, dpi=150)
    plt.close()


def main() -> None:
    criar_pastas_resultados()

    melhor_linha, modelo = carregar_melhor_modelo_otimizado()
    X_train, X_test, y_train, y_test = carregar_conjuntos()

    # Garante que o pipeline salvo está treinado no conjunto completo de treino.
    modelo.fit(X_train, y_train)

    y_pred, score_churn = gerar_predicoes(modelo, X_test)
    metricas = calcular_metricas(y_test, y_pred, score_churn)

    df_metricas = pd.DataFrame(
        [
            {
                "model": melhor_linha["model"],
                "balancing_scenario": melhor_linha["balancing_scenario"],
                "validation_best_score": melhor_linha["best_score"],
                **metricas,
            }
        ]
    )
    df_metricas.to_csv(ARQUIVO_METRICAS_FINAIS, index=False)

    salvar_predicoes(y_test, y_pred, score_churn)
    salvar_matriz_confusao(y_test, y_pred)
    salvar_curva_roc(y_test, score_churn)
    salvar_curva_precision_recall(y_test, score_churn)

    print("Avaliação final concluída.")
    print(df_metricas.to_string(index=False))
    print(f"\nMétricas salvas em: {ARQUIVO_METRICAS_FINAIS}")
    print(f"Predições salvas em: {ARQUIVO_PREDICOES_FINAIS}")
    print(f"Figuras salvas em: {PASTA_FIGURES}")


if __name__ == "__main__":
    main()
