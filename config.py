from pathlib import Path


# Semente usada para garantir reprodutibilidade nos experimentos.
SEMENTE_ALEATORIA = 42

# Pastas principais do projeto.
PASTA_DATA = Path("data")
PASTA_TREINO_TESTE = PASTA_DATA / "treino_teste"

# Pastas onde os resultados dos experimentos serão salvos.
PASTA_RESULTS = Path("results")
PASTA_METRICS = PASTA_RESULTS / "metrics"
PASTA_FIGURES = PASTA_RESULTS / "figures"
PASTA_MODELS = PASTA_RESULTS / "models"

# Arquivos gerados após a divisão entre treino e teste.
X_TRAIN_PATH = PASTA_TREINO_TESTE / "X_train.csv"
X_TEST_PATH = PASTA_TREINO_TESTE / "X_test.csv"
Y_TRAIN_PATH = PASTA_TREINO_TESTE / "y_train.csv"
Y_TEST_PATH = PASTA_TREINO_TESTE / "y_test.csv"

# Colunas com tratamento especial no pré-processamento.
COLUNA_TOTAL_CHARGES = "Total Charges"

# Quantidade de divisões usadas na validação cruzada.
N_SPLITS_CV = 5

# Métrica principal usada para comparar os modelos.
METRICA_PRINCIPAL = "f1_churn"

# Lista de pastas de resultados usadas pelos próximos scripts do projeto.
PASTAS_RESULTADOS = [
    PASTA_RESULTS,
    PASTA_METRICS,
    PASTA_FIGURES,
    PASTA_MODELS,
]


def criar_pastas_resultados() -> None:
    # Cria as pastas de resultados caso elas ainda não existam.
    for pasta in PASTAS_RESULTADOS:
        pasta.mkdir(parents=True, exist_ok=True)
