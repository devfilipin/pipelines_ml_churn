from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer

from config import PASTA_FIGURES, PASTA_METRICS, SEMENTE_ALEATORIA, criar_pastas_resultados
from preprocessamento import carregar_conjuntos


ARQUIVO_RESULTADOS_OTIMIZADOS = PASTA_METRICS / "optimized_results.csv"
ARQUIVO_IMPORTANCIA = PASTA_METRICS / "feature_importance.csv"
FIGURA_IMPORTANCIA = PASTA_FIGURES / "feature_importance.png"
FIGURA_SHAP = PASTA_FIGURES / "shap_summary.png"
TAMANHO_AMOSTRA = 300
N_REPEATS_PERMUTACAO = 5
TOP_FEATURES_GRAFICO = 20
MODELOS_ARVORES = {
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Random Forest",
    "Decision Tree",
}


def carregar_melhor_modelo() -> tuple[pd.Series, Any]:
    # Lê os resultados otimizados e carrega o melhor pipeline salvo.
    if not ARQUIVO_RESULTADOS_OTIMIZADOS.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_RESULTADOS_OTIMIZADOS}. "
            "Execute otimizar_modelos.py antes da interpretabilidade."
        )

    df_resultados = pd.read_csv(ARQUIVO_RESULTADOS_OTIMIZADOS)

    if df_resultados.empty:
        raise ValueError("O arquivo de resultados otimizados está vazio.")

    colunas_necessarias = ["model", "best_score", "model_path"]
    colunas_faltantes = [
        coluna for coluna in colunas_necessarias if coluna not in df_resultados.columns
    ]

    if colunas_faltantes:
        raise ValueError(
            "O arquivo de resultados otimizados não possui as colunas esperadas: "
            f"{colunas_faltantes}"
        )

    melhor_linha = df_resultados.sort_values("best_score", ascending=False).iloc[0]
    modelo = joblib.load(melhor_linha["model_path"])

    return melhor_linha, modelo


def obter_etapas_pipeline(pipeline: Any) -> tuple[Any, Any]:
    # Obtém o pré-processador e o modelo final do pipeline salvo.
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("O modelo salvo deve ser um pipeline com named_steps.")

    preprocessador = pipeline.named_steps.get("preprocessador")
    modelo_final = pipeline.named_steps.get("modelo")

    if preprocessador is None or modelo_final is None:
        raise ValueError("Pipeline sem as etapas esperadas: preprocessador e modelo.")

    return preprocessador, modelo_final


def obter_nomes_features(preprocessador: Any, X_train: pd.DataFrame) -> list[str]:
    # Usa os nomes gerados pelo pré-processamento quando disponíveis.
    try:
        return preprocessador.get_feature_names_out().tolist()
    except AttributeError:
        return X_train.columns.tolist()


def transformar_amostra(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Any, pd.DataFrame, pd.Series, list[str], Any]:
    # Garante que o pipeline está ajustado e transforma uma amostra do teste.
    preprocessador_salvo = pipeline.named_steps.get("preprocessador")
    if preprocessador_salvo is not None:
        try:
            preprocessador_salvo.set_output(transform="pandas")
        except AttributeError:
            pass

    pipeline.fit(X_train, y_train)
    preprocessador, modelo_final = obter_etapas_pipeline(pipeline)

    tamanho = min(TAMANHO_AMOSTRA, len(X_test))
    X_amostra = X_test.sample(
        n=tamanho,
        random_state=SEMENTE_ALEATORIA,
    )
    y_amostra = y_test.loc[X_amostra.index]
    X_transformado = preprocessador.transform(X_amostra)
    nomes_features = obter_nomes_features(preprocessador, X_train)

    if not isinstance(X_transformado, pd.DataFrame):
        X_transformado = pd.DataFrame(
            X_transformado,
            columns=nomes_features,
            index=X_amostra.index,
        )

    return X_transformado, X_amostra, y_amostra, nomes_features, modelo_final


def calcular_importancia_shap(
    model_name: str,
    modelo_final: Any,
    X_transformado: Any,
    nomes_features: list[str],
) -> pd.DataFrame | None:
    # Tenta usar SHAP para modelos baseados em árvores e gradient boosting.
    if model_name not in MODELOS_ARVORES:
        return None

    try:
        import shap

        explainer = shap.TreeExplainer(modelo_final)
        shap_values = explainer.shap_values(X_transformado)

        if isinstance(shap_values, list):
            valores = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif getattr(shap_values, "ndim", 0) == 3:
            valores = shap_values[:, :, 1]
        else:
            valores = shap_values

        importancia = abs(valores).mean(axis=0)

        shap.summary_plot(
            valores,
            X_transformado,
            feature_names=nomes_features,
            show=False,
            max_display=TOP_FEATURES_GRAFICO,
        )
        plt.title("Resumo SHAP - Melhor Modelo")
        plt.tight_layout()
        plt.savefig(FIGURA_SHAP, dpi=150, bbox_inches="tight")
        plt.close()

        return pd.DataFrame(
            {
                "feature": nomes_features,
                "importance": importancia,
                "method": "shap",
            }
        )
    except Exception as erro:
        print(f"SHAP não foi aplicado. Usando permutation_importance. Motivo: {erro}")
        return None


def calcular_importancia_permutacao(
    modelo_final: Any,
    X_transformado: Any,
    y_amostra: pd.Series,
    nomes_features: list[str],
) -> pd.DataFrame:
    # Fallback: calcula importância por permutação nas features transformadas.
    scorer = make_scorer(f1_score, pos_label=1, zero_division=0)
    resultado = permutation_importance(
        modelo_final,
        X_transformado,
        y_amostra,
        scoring=scorer,
        n_repeats=N_REPEATS_PERMUTACAO,
        random_state=SEMENTE_ALEATORIA,
        n_jobs=-1,
    )

    return pd.DataFrame(
        {
            "feature": nomes_features,
            "importance": resultado.importances_mean,
            "importance_std": resultado.importances_std,
            "method": "permutation_importance",
        }
    )


def salvar_grafico_importancia(df_importancia: pd.DataFrame) -> None:
    # Salva gráfico com os atributos mais importantes.
    df_plot = (
        df_importancia.sort_values("importance", ascending=False)
        .head(TOP_FEATURES_GRAFICO)
        .sort_values("importance")
    )

    plt.figure(figsize=(10, 8))
    plt.barh(df_plot["feature"], df_plot["importance"])
    plt.xlabel("Importância")
    plt.title("Principais atributos associados à previsão de churn")
    plt.tight_layout()
    plt.savefig(FIGURA_IMPORTANCIA, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    criar_pastas_resultados()

    melhor_linha, pipeline = carregar_melhor_modelo()
    X_train, X_test, y_train, y_test = carregar_conjuntos()
    X_transformado, _, y_amostra, nomes_features, modelo_final = transformar_amostra(
        pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    df_importancia = calcular_importancia_shap(
        model_name=melhor_linha["model"],
        modelo_final=modelo_final,
        X_transformado=X_transformado,
        nomes_features=nomes_features,
    )

    if df_importancia is None:
        df_importancia = calcular_importancia_permutacao(
            modelo_final=modelo_final,
            X_transformado=X_transformado,
            y_amostra=y_amostra,
            nomes_features=nomes_features,
        )

    df_importancia = df_importancia.sort_values("importance", ascending=False)
    df_importancia.to_csv(ARQUIVO_IMPORTANCIA, index=False)
    salvar_grafico_importancia(df_importancia)

    print("Interpretabilidade concluída.")
    print(df_importancia.head(20).to_string(index=False))
    print(f"\nImportâncias salvas em: {ARQUIVO_IMPORTANCIA}")
    print(f"Gráfico salvo em: {FIGURA_IMPORTANCIA}")
    if FIGURA_SHAP.exists():
        print(f"Resumo SHAP salvo em: {FIGURA_SHAP}")


if __name__ == "__main__":
    main()
