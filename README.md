# Bank Customer Churn Prediction

Projeto de Machine Learning para prever a rotatividade de clientes (churn) de um banco. O projeto cobre todo o fluxo de um problema de classificação desbalanceada: análise exploratória, escolha de uma métrica alinhada ao negócio, comparação de 9 combinações de modelos e técnicas de balanceamento com validação cruzada aninhada e avaliação final em um conjunto de teste isolado.

---

## Objetivo

Construir um modelo capaz de identificar, com antecedência, os clientes com maior risco de deixar o banco, permitindo que o time de retenção aja antes da saída.

Reter um cliente costuma ser muito mais barato do que adquirir um novo. Por isso, o foco do modelo é **não deixar clientes em risco passarem despercebidos** (recall alto), sem gerar alarmes falsos em excesso (precisão aceitável).

---

## Dataset

Fonte: [Bank Customer Churn (Kaggle)](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn).

O arquivo [`Customer-Churn-Records.csv`](Customer-Churn-Records.csv) contém **10.000 clientes** e **18 colunas**, sem valores nulos e sem registros duplicados. A variável alvo é desbalanceada: **~20,4% dos clientes deram churn**.

| Coluna | Tipo | Descrição |
|---|---|---|
| `RowNumber` | Numérico | Número do registro |
| `CustomerId` | Numérico | Identificador único do cliente (removida) |
| `Surname` | Texto | Sobrenome do cliente |
| `CreditScore` | Numérico | Pontuação de crédito |
| `Geography` | Categórico | País do cliente (`France`, `Spain`, `Germany`) |
| `Gender` | Categórico | Gênero do cliente |
| `Age` | Numérico | Idade |
| `Tenure` | Numérico | Anos de relacionamento com o banco |
| `Balance` | Numérico | Saldo em conta |
| `NumOfProducts` | Numérico | Quantidade de produtos contratados |
| `HasCrCard` | Binário | Possui cartão de crédito |
| `IsActiveMember` | Binário | É membro ativo |
| `EstimatedSalary` | Numérico | Salário estimado |
| `Complain` | Binário | Fez reclamação|
| `Satisfaction Score` | Numérico | Satisfação com a resolução da reclamação |
| `Card Type` | Categórico | Tipo de cartão (`SILVER`, `GOLD`, `PLATINUM`, `DIAMOND`) |
| `Point Earned` | Numérico | Pontos acumulados com o cartão de crédito |
| `Exited` | Binário | **Variável alvo**: 1 se o cliente saiu do banco, 0 caso contrário |

### Data Leakage: a variável `Complain`

O heatmap de correlação mostrou que `Complain` tem correlação praticamente perfeita (~1,0) com `Exited`. Na prática, a reclamação formal acontece junto com o processo de saída ou como consequência dele, então não é uma informação disponível *antes* do churn. Mantê-la não faria o modelo aprender a prever quem está prestes a sair do banco, mas sim olhar somente para a variável `Complain` na hora de classificar. Por isso, a coluna foi removida.

---

## Metodologia

### 1. Análise exploratória (EDA)

- **Univariada:** distribuições das variáveis categóricas e contínuas, além de boxplots para outliers. `Age` e `CreditScore` têm outliers, mas nada que prejudique modelos baseados em árvore. `Balance` tem uma grande concentração de contas zeradas.
- **Bivariada (contra `Exited`):**
  - `NumOfProducts`: clientes com 3 ou 4 produtos saem em sua grande maioria, e clientes com 2 produtos quase não saem.
  - `Geography`: clientes de `Germany` têm taxa de churn visivelmente maior que os de `France` e `Spain`.
  - `Age`: clientes mais velhos tendem a sair mais. É a única variável contínua com diferença clara entre as classes.

### 2. Preparação dos dados

- Split estratificado **80% treino / 20% teste** (`random_state=42`). O conjunto de teste só é usado na avaliação final.
- Pré-processamento com `ColumnTransformer`:
  - `OneHotEncoder` para `Geography`, `Gender` e `Card Type`;
  - `StandardScaler` para as variáveis numéricas.
- Todo o pré-processamento fica **dentro de uma `Pipeline`**, sendo reajustado a cada dobra da validação cruzada, o que evita vazamento de dados entre treino e validação.

### 3. Métrica de otimização: F2-score

Recall puro pode ser "trapaceado": um modelo que classifica todo mundo como churn tem recall de 100%. A métrica escolhida foi o **F2-score**, média harmônica entre precisão e recall com peso maior no recall:

$$F_\beta = (1 + \beta^2) \cdot \frac{precisão \cdot recall}{\beta^2 \cdot precisão + recall}, \quad \beta = 2$$

Ela traduz a regra de negócio: *perder um cliente custa mais caro do que oferecer um desconto desnecessário, mas não infinitamente mais caro.*

### 4. Baselines

Antes de otimizar, três referências foram avaliadas com `StratifiedKFold` (5 dobras):

| Modelo | F2 | Recall | Precisão |
|---|---|---|---|
| `DummyClassifier` (classe majoritária) | 0,000 | 0,000 | 0,000 |
| `LogisticRegression` | 0,238 | 0,207 | 0,598 |
| `DecisionTreeClassifier` (`max_depth=10`) | 0,494 | 0,471 | 0,616 |

O `DummyClassifier` acerta ~80% dos casos sem prever nenhum churn, o que mostra por que **acurácia não serve** como métrica prioritária aqui.

### 5. Modelagem com validação cruzada aninhada

Foram testados **3 algoritmos × 3 estratégias de balanceamento = 9 combinações**:

- Algoritmos: `DecisionTreeClassifier`, `RandomForestClassifier` e `LogisticRegression`;
- Balanceamento: `class_weight='balanced'` (como hiperparâmetro), oversampling com `SMOTE` e undersampling com `NearMiss` (versões 1, 2 e 3). As técnicas de amostragem rodam dentro de uma `imblearn.pipeline.Pipeline`, sendo aplicadas somente nos dados de treino de cada dobra.

Cada combinação foi otimizada com `RandomizedSearchCV` (refit por F2) em uma **validação cruzada aninhada**:

- **Loop interno** (`StratifiedKFold`, 3 dobras): busca de hiperparâmetros;
- **Loop externo** (`StratifiedKFold`, 5 dobras): estimativa honesta do desempenho do processo de busca.

### 6. Ranking (validação cruzada aninhada, conjunto de treino)

| Modelo | F2 | Desvio | Recall | Precisão | F1 |
|---|---|---|---|---|---|
| **Random Forest** | **0,670** | 0,009 | 0,715 | 0,535 | 0,612 |
| Random Forest + NearMiss | 0,651 | 0,011 | 0,707 | 0,494 | 0,582 |
| Random Forest + SMOTE | 0,639 | 0,020 | 0,651 | 0,594 | 0,621 |
| Decision Tree + NearMiss | 0,633 | 0,013 | 0,848 | 0,315 | 0,459 |
| Decision Tree | 0,630 | 0,014 | 0,712 | 0,433 | 0,537 |
| Decision Tree + SMOTE | 0,617 | 0,017 | 0,660 | 0,503 | 0,567 |
| Logistic Regression | 0,602 | 0,019 | 0,707 | 0,378 | 0,493 |
| Logistic Regression + SMOTE | 0,584 | 0,016 | 0,700 | 0,355 | 0,469 |
| Logistic Regression + NearMiss | 0,574 | 0,010 | 0,755 | 0,293 | 0,422 |

O **Random Forest com `class_weight='balanced'`** venceu com o maior F2 e o menor desvio entre dobras. Técnicas de amostragem não superaram o balanceamento por peso de classe. O `NearMiss` com Decision Tree, por exemplo, atinge recall de 0,848, mas derruba a precisão para 0,315.

Hiperparâmetros do modelo vencedor:

```
max_depth        = 8
min_samples_leaf = 20
max_features     = 0.5
class_weight     = balanced
```

---

## Métricas finais (conjunto de teste)

Avaliação do modelo vencedor nos **2.000 clientes** do conjunto de teste, nunca vistos durante o treino e a otimização:

| Métrica (classe churn) | Valor |
|---|---|
| **F2-score** (métrica-alvo) | **0,690** |
| Recall | 0,743 |
| Precisão | 0,536 |
| F1-score | 0,623 |
| Acurácia | 0,817 |

### Leitura de negócio

Dos 2.000 clientes de teste, 408 realmente saíram do banco:

- **303 clientes em risco foram identificados** (74,3% dos que sairiam), e o time de retenção teria a chance de agir.
- **105 clientes saíram sem nenhum alerta**. É o erro mais caro.
- **263 clientes receberiam contato desnecessário**. É o erro barato: custa a ação de retenção, não o cliente.

**Limitação:** com precisão de 53,6%, cerca de metade dos alertas é falso positivo. Aumentar a precisão exigiria aceitar um recall menor. Qual erro priorizar é uma decisão de negócio e depende da relação entre o custo de uma campanha de retenção e o valor de um cliente perdido.

---

## Como rodar

### Opção 1: Google Colab

1. Abra o notebook no Colab: [bank_customer_churn.ipynb](https://colab.research.google.com/github/ghfreiree/bank-customer-churn-prediction/blob/main/bank_customer_churn.ipynb)
2. Faça upload do arquivo `Customer-Churn-Records.csv` para a pasta `/content` (ícone de pasta na barra lateral).
3. Execute todas as células (*Ambiente de execução → Executar tudo*).

### Opção 2: Localmente

```bash
git clone https://github.com/ghfreiree/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter notebook bank_customer_churn.ipynb
```

O notebook lê o dataset por caminho relativo (`Customer-Churn-Records.csv`), então funciona sem alterações tanto no Colab quanto localmente, desde que o CSV esteja na mesma pasta do notebook.

> A validação cruzada aninhada treina centenas de modelos. A execução completa pode levar alguns minutos.

---

## Tecnologias utilizadas

- **Linguagem:** Python 3.13.15
- **Bibliotecas:** pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, Seaborn
- **Ambiente:** Google Colab / Jupyter Notebook
