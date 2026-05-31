# Pipelines de Machine Learning para Predição de Churn

Este repositório contém os códigos utilizados para carregar, preparar e dividir a base de dados **Telco Customer Churn** em conjuntos de treino e teste. O objetivo é organizar o pipeline inicial de experimentos de Machine Learning aplicados à predição de churn de clientes.

## Objetivo do projeto

O projeto tem como finalidade preparar a base de dados para experimentos de classificação, permitindo a construção posterior de modelos preditivos capazes de identificar clientes com maior probabilidade de churn.

Nesta etapa, o repositório contempla:

* carregamento da base de dados;
* tratamento inicial dos dados;
* separação entre variáveis preditoras e variável alvo;
* divisão da base em treino e teste;
* geração dos arquivos `X_train`, `X_test`, `y_train` e `y_test`.

## Estrutura do projeto

```text
.
├── base_telco.py
├── dividir_treino_teste.py
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── treino_teste/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── .gitignore
└── README.md
```

## Arquivos principais

### `base_telco.py`

Script responsável por carregar e preparar a base de dados Telco Customer Churn.

### `dividir_treino_teste.py`

Script responsável por dividir a base em conjuntos de treino e teste, gerando os arquivos separados para as variáveis independentes e para a variável alvo.

### `data/Telco-Customer-Churn.csv`

Base de dados original utilizada no projeto.

### `data/treino_teste/`

Pasta gerada após a execução do script de divisão da base. Ela contém os arquivos separados para treino e teste.

## Base de dados

A base utilizada neste projeto é a **Telco Customer Churn**, uma base pública amplamente utilizada em estudos de predição de churn.

Ela contém informações relacionadas ao perfil dos clientes, serviços contratados, dados de cobrança e a indicação de churn.

## Como executar o projeto

Primeiro, clone este repositório:

```bash
git clone https://github.com/devfilipin/pipelines_ml_churn.git
```

Acesse a pasta do projeto:

```bash
cd pipelines_ml_churn
```

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

Execute o script de preparação da base:

```bash
python base_telco.py
```

Depois, execute o script de divisão em treino e teste:

```bash
python dividir_treino_teste.py
```

Após a execução, os arquivos resultantes serão gerados na pasta:

```text
data/treino_teste/
```

## Reprodutibilidade

A divisão da base em treino e teste deve utilizar uma semente aleatória fixa, permitindo que os resultados possam ser reproduzidos em diferentes execuções.

## Tecnologias utilizadas

* Python
* Pandas
* Scikit-learn

## Status do projeto

Projeto em desenvolvimento.

Atualmente, o repositório contempla a etapa inicial do pipeline de Machine Learning, incluindo preparação e divisão da base de dados. As próximas etapas incluem treinamento, avaliação e comparação de modelos preditivos para churn.