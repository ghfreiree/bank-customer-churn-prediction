# Bank Customer Churn Prediction

Este é um estudo de caso de Machine Learning focado em prever a rotatividade de clientes (churn) em uma instituição financeira. O projeto é focado no desenvolvimento de um modelo de Machine Learning de classificação, desde testes com um modelo baseline até o tratamento de desbalanceamento de classes usando métodos de Ensemble e técnicas de validação cruzada.

## Visão Geral do Projeto
Reter clientes existentes é frequentemente muito mais econômico para as instituições financeiras do que adquirir novos. Este projeto constrói um pipeline preditivo robusto para identificar clientes com alto risco de deixar o banco (Churn), permitindo estratégias proativas de retenção.

O principal desafio técnico foi gerenciar um conjunto de dados altamente desbalanceado, mantendo um framework de validação matematicamente seguro para evitar vazamento de dados e métricas excessivamente otimistas ou tendenciosas.

---

## Dataset & Atributos
O conjunto de dados contém perfis demográficos e financeiros de clientes bancários.

* **RowNumber / CustomerId / Surname:** Identificadores únicos (removidos durante o pré-processamento por não possuírem poder preditivo).
* **CreditScore / Balance / EstimatedSalary:** Indicadores de saúde financeira.
* **Geography / Gender / Card Type:** Características demográficas categóricas.
* **Age:** Idade do cliente (altamente relevante para padrões de fidelidade).
* **Tenure:** Anos em que o cliente está com o banco.
* **NumOfProducts:** Quantidade de produtos adquiridos pelo cliente no banco.
* **HasCrCard / IsActiveMember:** Flags binárias de comportamento.
* **Points Earned:** Pontos acumulados pelo cliente ao utilizar o cartão de crédito.
* **Exited:** **Variável Alvo (Target)** (1 se o cliente deixou o banco, 0 caso contrário).

---

## Identificação de Data Leakage (Insight Chave de Engenharia)
Durante a análise de correlação via Heatmaps, um padrão crítico foi identificado: a variável **`Complain`** tinha uma correlação perfeita ($1.0$) com a variável alvo `Exited`.

* **O Problema:** Em um cenário real, o fato de um cliente registrar uma reclamação formal ocorre de forma praticamente simultânea ao processo de churn ou como uma consequência direta dele. Manter essa variável no conjunto de dados causa **Vazamento de Dados** — o modelo passa a depender fortemente dessa única "característica do futuro" e performa perfeitamente durante o treino, mas falha completamente ao generalizar para clientes novos na realidade.
* **A Solução:** O atributo `Complain` foi totalmente removido dos dados de treinamento, forçando os modelos a aprenderem os verdadeiros padrões comportamentais e financeiros subjacentes, em vez de dependerem de uma variável viciada.

---

## Metodologia & Evolução dos Modelos

Este projeto seguiu uma abordagem estruturada de benchmarking para encontrar a solução mais equilibrada e lucrativa para o banco:

### 1. Modelo Baseline: Árvore de Decisão (Decision Tree)
* **Abordagem:** Uma Árvore de Decisão pura, permitindo que ela crescesse sem restrições de profundidade.
* **Resultado:** Sofreu de um severo **Overfitting**. Alcançou quase 100% de acerto nos dados de treino, mas o desempenho despencou nos dados de avaliação, servindo apenas como um baseline instável.

### 2. Tratando o Desbalanceamento: Técnicas de Amostragem (NearMiss & SMOTE)
* **Abordagem:** Aplicação de Undersampling (NearMiss) e Oversampling (SMOTE) para balancear a classe alvo.
* **Resultado:** Embora o NearMiss tenha aumentado drasticamente o **Recall** (capturando quase todos os casos de churn), sua **Precisão** colapsou para a faixa de 24-25%. Para um banco, uma precisão tão baixa significa gerar uma quantidade massiva de alarmes falsos, desperdiçando orçamento de marketing com clientes que seriam fiéis de qualquer forma.

### 3. O Campeão: Random Forest com Ajuste de Pesos de Classe
* **Abordagem:** Implementação de um classificador **Random Forest** utilizando balanceamento algorítmico interno via parâmetro `class_weight='balanced'`.
* **Resultado:** Esta abordagem alcançou o equilíbrio de negócios ideal. Forneceu um Recall competitivo e estável, ao mesmo tempo em que recuperou significativamente a Precisão, resultando em um **F1-Score** muito mais alto e seguro para tomada de decisão.

---

## Framework de Validação Robusta
Para garantir a confiabilidade do modelo antes de qualquer decisão de negócio, um pipeline de validação rigoroso foi construído utilizando o `scikit-learn`:

* **`StratifiedKFold` (5 Splits):** Preservou a distribuição original da classe minoritária em todas as dobras de treino e validação.
* **`cross_validate`:** Utilizado para calcular os intervalos de confiança das métricas (Precisão, Recall e F1-score) ao longo das dobras, garantindo que o desempenho do modelo não seja fruto de uma divisão de dados "de sorte".
* **`cross_val_predict`:** Gerou previsões seguras e fora da dobra (*out-of-fold*) cobrindo todo o conjunto de treino. Isso permitiu uma avaliação das principais métricas do modelo, simulando exatamente como ele reagirá a dados novos em produção.

---

## Principais Aprendizados Técnicos
* **Métricas Estatísticas vs. Métricas de Negócio:** Maximizar o Recall destruindo a Precisão (como na tentativa com NearMiss) é financeiramente inviável para aplicações de negócios reais.
* **Organização de Estados no Notebook:** Prática de gerenciamento limpo de código ao redefinir e isolar explicitamente os objetos de cada modelo em suas respectivas células, garantindo outputs reprodutíveis, confiáveis e sem misturar o histórico de treinamento.

---

## Tecnologias Utilizadas
* **Linguagem:** Python 3.x
* **Bibliotecas:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn, Matplotlib, Seaborn
* **Ambiente:** Google Colab / Jupyter Notebook
