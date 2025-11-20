# Modelagem e Avaliação de Algoritmos: Eficiência Energética

Aplicação desenvolvida em **[Python 3.11](https://docs.python.org/pt-br/3.11/contents.html)** para realizar a **modelagem supervisionada de regressão** como parte da **Prova de Conceito (PoC)** de um sistema preditivo para **eficiência energética de edifícios**, com base no conjunto de dados público **[UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/energy+efficiency)**.

O objetivo deste projeto é **investigar, treinar e avaliar** diferentes algoritmos de regressão supervisionada capazes de prever:

- **Heating Load (Carga de aquecimento)**
- **Cooling Load (Carga de resfriamento)**

e comparar o desempenho entre eles por meio de métricas apropriadas.

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

## Objetivos do Projeto

- Pré-processar dados adequadamente
- Treinar diferentes técnicas de regressão supervisionada
- Avaliar o desempenho com métricas apropriadas
- Comparar modelos de forma tabular e visual
- Gerar relatório PDF
- Exportar modelo final e scaler para produção
- Gerar gráficos e logs da execução

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

## Funcionalidades

| Categoria | Descrição |
|------------|------------|
| **Pré-processamento** | Padronização (StandardScaler) e normalização (MinMaxScaler). |
| **Treinamento de modelos** | Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting (MultiOutput), SVR (MultiOutput). MultiOutput. |
| **Avaliação de Modelos** | RMSE, MAE e R². |
| **Comparação Gráfica** | Gráficos automáticos de comparação. |
| **Geração de Relatório PDF** | Compila métricas, gráficos e considerações em `outputs/relatorio_modelagem.pdf`. |
| **Exportação** | Modelo, scaler, colunas salvas e métricas em csv. |

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

## Tecnologias Utilizadas

| Categoria | Tecnologia / Biblioteca |
|------------|------------------------|
| Manipulação de Dados | **[pandas](https://pandas.pydata.org/docs/)**, **[numpy](https://numpy.org/doc/)** |
| Visualização | **[matplotlib](https://matplotlib.org/stable/users/explain/quick_start.html)**, **[seaborn](https://seaborn.pydata.org/)** |
| Machine Learning | **[scikit-learn](https://scikit-learn.org/stable/)** |
| Relatórios | **[reportlab](https://docs.reportlab.com/releases/notes/whats-new-40/)** |
| Utilitários | **[os](https://docs.python.org/pt-br/3/library/os.html)**, **[datetime](https://docs.python.org/pt-br/3/library/datetime.html)** |

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

## Estrutura de Diretórios

```
├── data/
│    └── ENB2012_data.xlsx
│
├── outputs/
│    ├── figs/
│    │    ├── comparacao_RMSE.png
│    │    ├── comparacao_MAE.png
│    │    └── comparacao_R2.png
│    │
│    ├── ml/
│    │    ├── modelo.pkl
│    │    ├── scaler.pkl
│    │    └── colunas_modelo.json
│    │
│    ├── metricas_modelos.csv
│    └── relatorio_modelagem.pdf
│    └── logs.txt
|
├── src/
│    ├── main.py
│    └── modelagem_regressao.py
│


```
**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

## Instalação e Execução

### Passo 1 – Criar o ambiente virtual
```bash
$ python -m venv .venv
$ source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
```

### Passo 2 – Instalar dependências
```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

### Passo 3 – Execute o Projeto
```bash
$ python src/main.py
```


**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
## Relatório e Métricas
Após a execução serão gerados:

- Relatório PDF
- CSV com as métricas
- Gráficos comparativos
- Modelo e scaler final para produção

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
## Artefatos para Produção

| Arquivo    | Finalidade |
|------------|------------|
| `modelo.pkl` | Modelo final escolhido com melhor R² |
| `scaler.pkl` | Padronização necessária para predições |
|`colunas_modelo.json` | Ordem correta das features |
---
### Integração com aplicação Django:

Utilizará os arquivos gerados:

- `modelo.pkl` → para executar predições reais
- `scaler.pkl` → para pré-processar entradas do usuário
- `colunas_modelo.json` → para manter a ordem correta das features

Isso garante que o sistema web fará predições exatamente como o modelo foi treinado.