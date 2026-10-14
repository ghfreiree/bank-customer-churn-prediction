"""Configurações compartilhadas pelos notebooks: semente e caminhos do projeto."""
from pathlib import Path

RANDOM_STATE = 42   # semente única para todo o projeto, torna os resultados reprodutíveis

RAIZ = Path(__file__).resolve().parent.parent   # pasta raiz do projeto, independente de onde o notebook roda

DADOS_BRUTOS = RAIZ / 'data' / 'raw' / 'Customer-Churn-Records.csv'
DADOS_TRATADOS = RAIZ / 'data' / 'processed' / 'Customer-Churn-Tratado.csv'   # gerado no notebook 01_eda
MODELO_FINAL = RAIZ / 'models' / 'modelo_churn.pkl'                            # gerado no notebook 02_modelagem
PASTA_FIGURAS = RAIZ / 'reports' / 'figures'                                   # figuras usadas no README
