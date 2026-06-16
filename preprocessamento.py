import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    COLUNA_TOTAL_CHARGES,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
)


def converter_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    # Garante que Total Charges chegue como número ao ColumnTransformer.
    if COLUNA_TOTAL_CHARGES not in df.columns:
        raise ValueError(f"A coluna '{COLUNA_TOTAL_CHARGES}' não existe na base.")

    df = df.copy()
    df[COLUNA_TOTAL_CHARGES] = pd.to_numeric(
        df[COLUNA_TOTAL_CHARGES].astype(str).str.strip().replace("", pd.NA),
        errors="coerce",
    )

    if not pd.api.types.is_numeric_dtype(df[COLUNA_TOTAL_CHARGES]):
        raise ValueError(f"A coluna '{COLUNA_TOTAL_CHARGES}' não está numérica.")

    return df


def carregar_conjuntos() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Carrega os conjuntos já separados entre treino e teste.
    x_train = converter_total_charges(pd.read_csv(X_TRAIN_PATH))
    x_test = converter_total_charges(pd.read_csv(X_TEST_PATH))
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")
    y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

    return x_train, x_test, y_train, y_test


def identificar_colunas(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    # Colunas numéricas são identificadas pelo tipo; as demais são categóricas.
    x = converter_total_charges(x)
    colunas_numericas = x.select_dtypes(include="number").columns.tolist()
    colunas_categoricas = x.select_dtypes(exclude="number").columns.tolist()

    if not pd.api.types.is_numeric_dtype(x[COLUNA_TOTAL_CHARGES]):
        raise ValueError(f"A coluna '{COLUNA_TOTAL_CHARGES}' não está numérica.")
    if COLUNA_TOTAL_CHARGES not in colunas_numericas:
        raise ValueError(f"A coluna '{COLUNA_TOTAL_CHARGES}' não está nas numéricas.")
    if COLUNA_TOTAL_CHARGES in colunas_categoricas:
        raise ValueError(f"A coluna '{COLUNA_TOTAL_CHARGES}' está nas categóricas.")

    return colunas_numericas, colunas_categoricas


def criar_one_hot_encoder() -> OneHotEncoder:
    # sparse_output é usado nas versões recentes do scikit-learn.
    # sparse mantém compatibilidade com versões mais antigas.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def criar_preprocessador(x: pd.DataFrame) -> ColumnTransformer:
    # Cria o pré-processador sem aplicar fit, para evitar vazamento de dados.
    colunas_numericas, colunas_categoricas = identificar_colunas(x)

    pipeline_numerico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline_categorico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", criar_one_hot_encoder()),
        ]
    )

    preprocessador = ColumnTransformer(
        transformers=[
            ("num", pipeline_numerico, colunas_numericas),
            ("cat", pipeline_categorico, colunas_categoricas),
        ]
    )

    # Mantém nomes das features transformadas para modelos como LightGBM.
    try:
        preprocessador.set_output(transform="pandas")
    except AttributeError:
        pass

    return preprocessador
