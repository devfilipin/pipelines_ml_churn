# Predição de Customer Churn em Telecomunicações

Este projeto desenvolve um fluxo experimental de machine learning para prever **customer churn** no setor de telecomunicações, usando o **Telco Customer Churn Dataset da IBM**.

O objetivo é comparar diferentes modelos supervisionados, avaliar cenários de balanceamento de classes, otimizar os melhores candidatos e identificar o modelo com melhor desempenho para prever clientes com maior risco de cancelamento.

## Estrutura de Pastas

```text
.
├── base_telco.py
├── dividir_treino_teste.py
├── preprocessamento.py
├── treinar_baselines.py
├── selecionar_melhores.py
├── otimizar_modelos.py
├── avaliar_final.py
├── interpretar_modelo.py
├── config.py
├── requirements.txt
├── data/
│   ├── CustomerChurn.xlsx
│   ├── Telco_customer_churn_demographics.xlsx
│   ├── Telco-Customer-Churn.csv
│   └── treino_teste/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
└── results/
    ├── metrics/
    ├── figures/
    └── models/
```

## Ordem de Execução

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute os scripts nesta ordem:

```bash
python base_telco.py
python dividir_treino_teste.py
python treinar_baselines.py
python selecionar_melhores.py
python otimizar_modelos.py
python avaliar_final.py
python interpretar_modelo.py
```

## Descrição dos Scripts

### `base_telco.py`

Consolida as bases originais em `data/Telco-Customer-Churn.csv`, juntando os dados principais de churn com informações demográficas.

### `dividir_treino_teste.py`

Divide a base consolidada em treino e teste, separando variáveis preditoras e variável alvo.

Arquivos gerados:

```text
data/treino_teste/X_train.csv
data/treino_teste/X_test.csv
data/treino_teste/y_train.csv
data/treino_teste/y_test.csv
```

### `preprocessamento.py`

Define funções reutilizáveis para carregar os conjuntos de treino e teste, identificar colunas numéricas e categóricas, e criar o pré-processador usado dentro dos pipelines.

### `treinar_baselines.py`

Treina e avalia modelos baseline usando validação cruzada estratificada em diferentes cenários de balanceamento:

* sem balanceamento;
* pesos de classe;
* SMOTE;
* SMOTE-Tomek;
* SMOTE-ENN.

Arquivo gerado:

```text
results/metrics/baseline_results.csv
```

### `selecionar_melhores.py`

Seleciona os 3 melhores candidatos para otimização de hiperparâmetros com base nas métricas de validação cruzada.

Arquivo gerado:

```text
results/metrics/selected_models.csv
```

### `otimizar_modelos.py`

Executa `RandomizedSearchCV` nos modelos selecionados, usando apenas o conjunto de treino.

Arquivos gerados:

```text
results/metrics/optimized_results.csv
results/metrics/best_params.csv
results/models/*.joblib
```

### `avaliar_final.py`

Seleciona o melhor modelo otimizado e realiza a avaliação final no conjunto de teste independente.

Arquivos gerados:

```text
results/metrics/final_test_metrics.csv
results/metrics/final_predictions.csv
results/figures/confusion_matrix_best_model.png
results/figures/roc_curve_best_model.png
results/figures/pr_curve_best_model.png
```

### `interpretar_modelo.py`

Gera análise de interpretabilidade do melhor modelo, usando SHAP quando possível e importância por permutação como alternativa.

Arquivos gerados:

```text
results/metrics/feature_importance.csv
results/figures/feature_importance.png
results/figures/shap_summary.png
```

## Arquivos Gerados

### Métricas

Os arquivos de métricas são salvos em:

```text
results/metrics/
```

Essa pasta contém resultados de validação cruzada, modelos selecionados, melhores parâmetros, métricas finais no teste e predições finais.

### Figuras

As figuras são salvas em:

```text
results/figures/
```

Essa pasta contém matriz de confusão, curva ROC, curva Precision-Recall e gráficos de interpretabilidade.

### Modelos Salvos

Os modelos otimizados são salvos em:

```text
results/models/
```

Cada modelo é salvo em formato `.joblib`.

## Observações Metodológicas

O conjunto de teste é usado somente na etapa de avaliação final, em `avaliar_final.py`. As etapas de baseline, seleção e otimização usam apenas dados de treino e validação cruzada.

O balanceamento de classes é aplicado somente dentro dos pipelines de treinamento. Isso evita vazamento de dados entre treino e validação.

A métrica principal do projeto é o **F1-score da classe churn**, pois o problema envolve uma classe positiva de maior interesse e desbalanceamento entre clientes que cancelam e não cancelam.

Além do F1-score, também são analisados **recall da classe churn** e **PR-AUC**. Essas métricas são importantes porque falsos negativos podem ter custo alto: um cliente propenso a churn pode não ser identificado a tempo para uma ação de retenção.

## Dependências Principais

* pandas
* scikit-learn
* imbalanced-learn
* xgboost
* lightgbm
* catboost
* shap
* matplotlib
* joblib

## Reprodutibilidade

A semente aleatória do projeto é definida em `config.py`:

```python
SEMENTE_ALEATORIA = 42
```

Essa configuração é usada na divisão treino/teste, validação cruzada, balanceamento e modelos sempre que aplicável.
