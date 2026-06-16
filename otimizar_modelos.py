from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any
import json

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None
    SMOTETomek = None
    SMOTEENN = None
    ImbPipeline = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from config import (
    N_SPLITS_CV,
    PASTA_METRICS,
    PASTA_MODELS,
    SEMENTE_ALEATORIA,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    criar_pastas_resultados,
)
from preprocessamento import converter_total_charges, criar_preprocessador


ARQUIVO_SELECIONADOS = PASTA_METRICS / "selected_models.csv"
ARQUIVO_RESULTADOS_OTIMIZADOS = PASTA_METRICS / "optimized_results.csv"
ARQUIVO_MELHORES_PARAMETROS = PASTA_METRICS / "best_params.csv"


def carregar_modelos_selecionados() -> pd.DataFrame:
    # Carrega os candidatos selecionados a partir dos resultados dos baselines.
    if not ARQUIVO_SELECIONADOS.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_SELECIONADOS}. "
            "Execute selecionar_melhores.py antes da otimização."
        )

    df_selecionados = pd.read_csv(ARQUIVO_SELECIONADOS)
    colunas_necessarias = ["model", "balancing_scenario"]
    colunas_faltantes = [
        coluna for coluna in colunas_necessarias if coluna not in df_selecionados.columns
    ]

    if colunas_faltantes:
        raise ValueError(
            "O arquivo de modelos selecionados não possui as colunas esperadas: "
            f"{colunas_faltantes}"
        )

    return df_selecionados


def carregar_treino() -> tuple[pd.DataFrame, pd.Series]:
    # Carrega apenas treino para evitar uso indevido do conjunto de teste.
    x_train = converter_total_charges(pd.read_csv(X_TRAIN_PATH))
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")

    return x_train, y_train


def calcular_peso_churn(y_train: pd.Series) -> float:
    # Peso usado por modelos que não aceitam class_weight="balanced".
    n_negativos = (y_train == 0).sum()
    n_positivos = (y_train == 1).sum()

    if n_positivos == 0:
        raise ValueError("y_train não possui exemplos da classe churn.")

    return float(n_negativos / n_positivos)


def criar_modelo(
    model_name: str,
    balancing_scenario: str,
    y_train: pd.Series,
) -> BaseEstimator:
    # Recria o modelo selecionado com random_state fixo quando aplicável.
    usar_pesos = balancing_scenario == "class_weight"
    class_weight = "balanced" if usar_pesos else None
    peso_churn = calcular_peso_churn(y_train)

    if model_name == "Logistic Regression":
        return LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=SEMENTE_ALEATORIA,
        )

    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=SEMENTE_ALEATORIA,
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            class_weight=class_weight,
            random_state=SEMENTE_ALEATORIA,
            n_jobs=-1,
        )

    if model_name == "Support Vector Machine":
        return SVC(
            class_weight=class_weight,
            probability=True,
            random_state=SEMENTE_ALEATORIA,
        )

    if model_name == "XGBoost":
        if XGBClassifier is None:
            raise ImportError("Instale xgboost para otimizar o modelo XGBoost.")

        return XGBClassifier(
            eval_metric="logloss",
            random_state=SEMENTE_ALEATORIA,
            scale_pos_weight=peso_churn if usar_pesos else 1,
            n_jobs=-1,
        )

    if model_name == "LightGBM":
        if LGBMClassifier is None:
            raise ImportError("Instale lightgbm para otimizar o modelo LightGBM.")

        return LGBMClassifier(
            class_weight=class_weight,
            random_state=SEMENTE_ALEATORIA,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "CatBoost":
        if CatBoostClassifier is None:
            raise ImportError("Instale catboost para otimizar o modelo CatBoost.")

        return CatBoostClassifier(
            class_weights=(1.0, peso_churn) if usar_pesos else None,
            random_state=SEMENTE_ALEATORIA,
            verbose=0,
        )

    raise ValueError(f"Modelo não reconhecido: {model_name}")


def criar_parametros_busca(model_name: str) -> dict[str, list[Any]] | list[dict[str, list[Any]]]:
    # Espaços moderados de hiperparâmetros para manter o experimento viável.
    if model_name == "Logistic Regression":
        return [
            {
                "modelo__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "modelo__l1_ratio": [0.0],
                "modelo__solver": ["liblinear"],
            },
            {
                "modelo__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "modelo__l1_ratio": [1.0],
                "modelo__solver": ["liblinear"],
            },
        ]

    if model_name == "Decision Tree":
        return {
            "modelo__max_depth": [None, 3, 5, 8, 12, 16],
            "modelo__min_samples_split": [2, 5, 10, 20],
            "modelo__min_samples_leaf": [1, 2, 5, 10],
            "modelo__criterion": ["gini", "entropy"],
        }

    if model_name == "Random Forest":
        return {
            "modelo__n_estimators": [150, 300, 500],
            "modelo__max_depth": [None, 6, 10, 14],
            "modelo__min_samples_split": [2, 5, 10],
            "modelo__min_samples_leaf": [1, 2, 4],
            "modelo__max_features": ["sqrt", "log2"],
        }

    if model_name == "Support Vector Machine":
        return {
            "modelo__C": [0.1, 1.0, 5.0, 10.0],
            "modelo__kernel": ["linear", "rbf"],
            "modelo__gamma": ["scale", "auto", 0.01, 0.1],
        }

    if model_name == "XGBoost":
        return {
            "modelo__n_estimators": [150, 300, 500],
            "modelo__max_depth": [3, 4, 5, 6],
            "modelo__learning_rate": [0.01, 0.05, 0.1],
            "modelo__subsample": [0.8, 1.0],
            "modelo__colsample_bytree": [0.8, 1.0],
            "modelo__min_child_weight": [1, 3, 5],
        }

    if model_name == "LightGBM":
        return {
            "modelo__n_estimators": [150, 300, 500],
            "modelo__num_leaves": [15, 31, 63],
            "modelo__learning_rate": [0.01, 0.05, 0.1],
            "modelo__max_depth": [-1, 5, 10],
            "modelo__subsample": [0.8, 1.0],
            "modelo__colsample_bytree": [0.8, 1.0],
        }

    if model_name == "CatBoost":
        return {
            "modelo__iterations": [150, 300, 500],
            "modelo__depth": [4, 6, 8],
            "modelo__learning_rate": [0.01, 0.05, 0.1],
            "modelo__l2_leaf_reg": [1, 3, 5, 7],
        }

    raise ValueError(f"Modelo não reconhecido: {model_name}")


def criar_balanceador(balancing_scenario: str) -> Any | None:
    # Recria o cenário de balanceamento escolhido, sempre dentro do pipeline.
    if balancing_scenario in {"none", "class_weight"}:
        return None

    if ImbPipeline is None:
        raise ImportError(
            "Instale imbalanced-learn para usar SMOTE, SMOTE-Tomek ou SMOTE-ENN."
        )

    if balancing_scenario == "smote":
        return SMOTE(random_state=SEMENTE_ALEATORIA)

    if balancing_scenario == "smote_tomek":
        return SMOTETomek(random_state=SEMENTE_ALEATORIA)

    if balancing_scenario == "smote_enn":
        return SMOTEENN(random_state=SEMENTE_ALEATORIA)

    raise ValueError(f"Cenário de balanceamento não reconhecido: {balancing_scenario}")


def criar_pipeline(
    model_name: str,
    balancing_scenario: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> SklearnPipeline | Any:
    # Recria pré-processador, balanceador e modelo para o candidato selecionado.
    preprocessador = criar_preprocessador(X_train)
    balanceador = criar_balanceador(balancing_scenario)
    modelo = criar_modelo(model_name, balancing_scenario, y_train)

    if balanceador is None:
        return SklearnPipeline(
            steps=[
                ("preprocessador", preprocessador),
                ("modelo", modelo),
            ]
        )

    return ImbPipeline(
        steps=[
            ("preprocessador", preprocessador),
            ("balanceador", balanceador),
            ("modelo", modelo),
        ]
    )


def calcular_n_iter(
    parametros_busca: dict[str, list[Any]] | list[dict[str, list[Any]]],
    n_iter_padrao: int = 30,
) -> int:
    # Evita pedir mais iterações do que combinações disponíveis.
    if isinstance(parametros_busca, list):
        total_combinacoes = sum(
            reduce(mul, (len(valores) for valores in grade.values()), 1)
            for grade in parametros_busca
        )
    else:
        total_combinacoes = reduce(
            mul,
            (len(valores) for valores in parametros_busca.values()),
            1,
        )

    return min(n_iter_padrao, total_combinacoes)


def criar_nome_arquivo_modelo(model_name: str, balancing_scenario: str) -> Path:
    # Cria nomes simples e estáveis para os modelos otimizados.
    nome_modelo = model_name.lower().replace(" ", "_")
    return PASTA_MODELS / f"{nome_modelo}_{balancing_scenario}.joblib"


def otimizar_modelo(
    model_name: str,
    balancing_scenario: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    n_iter: int | None = None,
) -> dict[str, Any]:
    # Executa a busca usando apenas dados de treino e validação cruzada.
    pipeline = criar_pipeline(model_name, balancing_scenario, X_train, y_train)
    parametros_busca = criar_parametros_busca(model_name)
    scorer = make_scorer(f1_score, pos_label=1, zero_division=0)
    n_iter_busca = n_iter or calcular_n_iter(parametros_busca)

    busca = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parametros_busca,
        n_iter=n_iter_busca,
        scoring=scorer,
        cv=cv,
        random_state=SEMENTE_ALEATORIA,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    busca.fit(X_train, y_train)

    caminho_modelo = criar_nome_arquivo_modelo(model_name, balancing_scenario)
    joblib.dump(busca.best_estimator_, caminho_modelo)

    return {
        "model": model_name,
        "balancing_scenario": balancing_scenario,
        "best_score": busca.best_score_,
        "best_params": json.dumps(busca.best_params_, ensure_ascii=False, default=str),
        "model_path": str(caminho_modelo),
    }


def main() -> None:
    criar_pastas_resultados()

    X_train, y_train = carregar_treino()
    modelos_selecionados = carregar_modelos_selecionados()
    cv = StratifiedKFold(
        n_splits=N_SPLITS_CV,
        shuffle=True,
        random_state=SEMENTE_ALEATORIA,
    )

    resultados = []

    for _, candidato in modelos_selecionados.iterrows():
        model_name = candidato["model"]
        balancing_scenario = candidato["balancing_scenario"]
        print(f"Otimizando {model_name} | {balancing_scenario}")

        resultado = otimizar_modelo(
            model_name=model_name,
            balancing_scenario=balancing_scenario,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
        )
        resultados.append(resultado)

    df_resultados = pd.DataFrame(resultados).sort_values(
        by="best_score",
        ascending=False,
    )
    df_resultados.to_csv(ARQUIVO_RESULTADOS_OTIMIZADOS, index=False)
    df_resultados.to_csv(ARQUIVO_MELHORES_PARAMETROS, index=False)

    print(f"\nResultados salvos em: {ARQUIVO_RESULTADOS_OTIMIZADOS}")
    print(f"Melhores parâmetros salvos em: {ARQUIVO_MELHORES_PARAMETROS}")
    print(df_resultados.to_string(index=False))


if __name__ == "__main__":
    main()
