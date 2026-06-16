from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline

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
    METRICA_PRINCIPAL,
    N_SPLITS_CV,
    PASTA_METRICS,
    SEMENTE_ALEATORIA,
    criar_pastas_resultados,
)
from preprocessamento import carregar_conjuntos, criar_preprocessador


ARQUIVO_RESULTADOS = PASTA_METRICS / "baseline_results.csv"


def listar_dependencias_faltantes() -> list[str]:
    # Informa bibliotecas ausentes para evitar resultados incompletos sem aviso.
    dependencias_faltantes = []

    if ImbPipeline is None:
        dependencias_faltantes.append("imbalanced-learn")
    if XGBClassifier is None:
        dependencias_faltantes.append("xgboost")
    if LGBMClassifier is None:
        dependencias_faltantes.append("lightgbm")
    if CatBoostClassifier is None:
        dependencias_faltantes.append("catboost")

    return dependencias_faltantes


def criar_modelos() -> dict[str, BaseEstimator]:
    # Modelos baseline sem ajuste específico para desbalanceamento.
    modelos: dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=SEMENTE_ALEATORIA,
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=SEMENTE_ALEATORIA),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=SEMENTE_ALEATORIA,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            probability=True,
            random_state=SEMENTE_ALEATORIA,
        ),
    }

    if XGBClassifier is not None:
        modelos["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=SEMENTE_ALEATORIA,
            n_jobs=-1,
        )

    if LGBMClassifier is not None:
        modelos["LightGBM"] = LGBMClassifier(
            random_state=SEMENTE_ALEATORIA,
            n_jobs=-1,
            verbose=-1,
        )

    if CatBoostClassifier is not None:
        modelos["CatBoost"] = CatBoostClassifier(
            random_state=SEMENTE_ALEATORIA,
            verbose=0,
        )

    return modelos


def criar_modelos_com_pesos(y_train: pd.Series) -> dict[str, BaseEstimator]:
    # Modelos com pesos de classe quando o algoritmo oferece esse recurso.
    n_negativos = (y_train == 0).sum()
    n_positivos = (y_train == 1).sum()
    peso_churn = float(n_negativos / n_positivos)

    modelos: dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEMENTE_ALEATORIA,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=SEMENTE_ALEATORIA,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            random_state=SEMENTE_ALEATORIA,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            class_weight="balanced",
            probability=True,
            random_state=SEMENTE_ALEATORIA,
        ),
    }

    if XGBClassifier is not None:
        modelos["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=SEMENTE_ALEATORIA,
            scale_pos_weight=peso_churn,
            n_jobs=-1,
        )

    if LGBMClassifier is not None:
        modelos["LightGBM"] = LGBMClassifier(
            class_weight="balanced",
            random_state=SEMENTE_ALEATORIA,
            n_jobs=-1,
            verbose=-1,
        )

    if CatBoostClassifier is not None:
        modelos["CatBoost"] = CatBoostClassifier(
            class_weights=(1.0, peso_churn),
            random_state=SEMENTE_ALEATORIA,
            verbose=0,
        )

    return modelos


def criar_cenarios_balanceamento() -> dict[str, Callable[[], Any] | None]:
    # O objeto de balanceamento é criado sob demanda para cada pipeline.
    return {
        "none": None,
        "class_weight": None,
        "smote": lambda: SMOTE(random_state=SEMENTE_ALEATORIA),
        "smote_tomek": lambda: SMOTETomek(random_state=SEMENTE_ALEATORIA),
        "smote_enn": lambda: SMOTEENN(random_state=SEMENTE_ALEATORIA),
    }


def criar_metricas() -> dict[str, str | Callable[..., float]]:
    # Métricas avaliadas na validação cruzada, com foco na classe churn.
    return {
        "accuracy": "accuracy",
        "precision_churn": make_scorer(
            precision_score_churn,
            pos_label=1,
            zero_division=0,
        ),
        "recall_churn": make_scorer(
            recall_score_churn,
            pos_label=1,
            zero_division=0,
        ),
        "f1_churn": make_scorer(
            f1_score_churn,
            pos_label=1,
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }


def precision_score_churn(
    y_true: pd.Series,
    y_pred: pd.Series,
    pos_label: int = 1,
    zero_division: int = 0,
) -> float:
    from sklearn.metrics import precision_score

    return precision_score(
        y_true,
        y_pred,
        pos_label=pos_label,
        zero_division=zero_division,
    )


def recall_score_churn(
    y_true: pd.Series,
    y_pred: pd.Series,
    pos_label: int = 1,
    zero_division: int = 0,
) -> float:
    from sklearn.metrics import recall_score

    return recall_score(
        y_true,
        y_pred,
        pos_label=pos_label,
        zero_division=zero_division,
    )


def f1_score_churn(
    y_true: pd.Series,
    y_pred: pd.Series,
    pos_label: int = 1,
    zero_division: int = 0,
) -> float:
    from sklearn.metrics import f1_score

    return f1_score(
        y_true,
        y_pred,
        pos_label=pos_label,
        zero_division=zero_division,
    )


def criar_pipeline(
    x_train: pd.DataFrame,
    modelo: BaseEstimator,
    balanceador: Any | None = None,
) -> SklearnPipeline | Any:
    # O pré-processador fica dentro do pipeline para evitar vazamento de dados.
    preprocessador = criar_preprocessador(x_train)

    if balanceador is None:
        return SklearnPipeline(
            steps=[
                ("preprocessador", preprocessador),
                ("modelo", modelo),
            ]
        )

    if ImbPipeline is None:
        raise ImportError(
            "Instale imbalanced-learn para usar cenários com balanceamento."
        )

    return ImbPipeline(
        steps=[
            ("preprocessador", preprocessador),
            ("balanceador", balanceador),
            ("modelo", modelo),
        ]
    )


def avaliar_modelo(
    nome_modelo: str,
    nome_cenario: str,
    modelo: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    metricas: dict[str, str | Callable[..., float]],
    balanceador: Any | None = None,
) -> dict[str, Any]:
    # A validação cruzada ajusta pré-processamento e balanceamento apenas no treino.
    pipeline = criar_pipeline(x_train, modelo, balanceador)
    resultados_cv = cross_validate(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring=metricas,
        n_jobs=1,
        error_score="raise",
    )

    resultado: dict[str, Any] = {
        "modelo": nome_modelo,
        "cenario_balanceamento": nome_cenario,
    }

    for metrica in metricas:
        valores = resultados_cv[f"test_{metrica}"]
        resultado[f"{metrica}_mean"] = valores.mean()
        resultado[f"{metrica}_std"] = valores.std()

    return resultado


def main() -> None:
    criar_pastas_resultados()

    dependencias_faltantes = listar_dependencias_faltantes()
    if dependencias_faltantes:
        print(
            "Aviso: algumas dependências não estão instaladas e serão ignoradas: "
            f"{', '.join(dependencias_faltantes)}"
        )

    x_train, _, y_train, _ = carregar_conjuntos()
    cv = StratifiedKFold(
        n_splits=N_SPLITS_CV,
        shuffle=True,
        random_state=SEMENTE_ALEATORIA,
    )
    metricas = criar_metricas()
    cenarios = criar_cenarios_balanceamento()

    resultados: list[dict[str, Any]] = []

    for nome_cenario, criar_balanceador in cenarios.items():
        if nome_cenario == "class_weight":
            modelos = criar_modelos_com_pesos(y_train)
        else:
            modelos = criar_modelos()

        for nome_modelo, modelo in modelos.items():
            if criar_balanceador is not None and ImbPipeline is None:
                continue

            balanceador = criar_balanceador() if criar_balanceador else None
            print(f"Avaliando {nome_modelo} | {nome_cenario}")

            resultado = avaliar_modelo(
                nome_modelo=nome_modelo,
                nome_cenario=nome_cenario,
                modelo=modelo,
                x_train=x_train,
                y_train=y_train,
                cv=cv,
                metricas=metricas,
                balanceador=balanceador,
            )
            resultados.append(resultado)

    df_resultados = pd.DataFrame(resultados).sort_values(
        by=[
            f"{METRICA_PRINCIPAL}_mean",
            "recall_churn_mean",
            "pr_auc_mean",
        ],
        ascending=False,
    )
    df_resultados.to_csv(ARQUIVO_RESULTADOS, index=False)

    print(f"\nResultados salvos em: {ARQUIVO_RESULTADOS}")
    print(df_resultados.head(10))


if __name__ == "__main__":
    main()
