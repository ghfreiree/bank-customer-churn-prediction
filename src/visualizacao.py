"""Padrão visual e funções de gráfico usadas nos notebooks."""
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Padrão visual único para todas as figuras do projeto
COR_PRINCIPAL = '#2a78d6'                       # azul: gráficos de uma única série
PALETA_CHURN = {0: '#2a78d6', 1: '#eb6834'}     # azul = não churn, laranja = churn (mesmas cores em todas as figuras)
LEGENDA_CHURN = ['Não churn', 'Churn']
CMAP_CORRELACAO = LinearSegmentedColormap.from_list('correlacao', ['#2a78d6', '#f0efec', '#e34948'])  # negativo azul, zero cinza, positivo vermelho
COR_REFERENCIA = '#52514e'   # cinza para textos e linhas de referência

PALETA_CATEGORIAS = ['#2a78d6', '#1baf7a', '#4a3aa7', '#eda100', '#e87ba4']   # sem laranja, que fica reservado para churn
CORES_TEXTO_ESCURO = {'#1baf7a', '#eda100', '#e87ba4'}                        # segmentos claros recebem texto escuro


def aplicar_tema():     # Tema do seaborn usado em todos os notebooks
    sns.set_theme(style='whitegrid', rc={
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.edgecolor': '#d0d0d0',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': '#ebebeb',
    })


def format_k(x, pos):     # Formata os eixos na casa dos milhares: 1500 -> 1.5k
    if abs(x) >= 1000:
        return f'{x*1e-3:g}k'
    return f'{x:.0f}'


k_formatter = ticker.FuncFormatter(format_k)


def rotular_barras(ax, fmt='{:.0f}', fontsize=10):     # Escreve o valor em cima de cada coluna (fmt: texto de formatação ou função)
    for container in ax.containers:
        rotulos = [fmt(b.get_height()) if callable(fmt) else fmt.format(b.get_height()) for b in container]
        ax.bar_label(container, labels=rotulos, padding=3, fontsize=fontsize, color=COR_REFERENCIA)
    ax.margins(y=0.12)   # Espaço extra no topo para o número não encostar na borda


def linha_referencia(ax, valor, texto):     # Linha tracejada com o valor da base toda, para comparação
    ax.axhline(valor, color=COR_REFERENCIA, linestyle='--', linewidth=1)
    ax.text(1.01, valor, texto, transform=ax.get_yaxis_transform(), va='center', fontsize=9, color=COR_REFERENCIA)


def barras_100(ax, dados, grupo, col, cores, nomes=None, rotulo_grupo=str):
    """Barras 100% empilhadas: como os clientes de cada categoria de `grupo` se dividem entre as categorias de `col`.
    A última barra ("Base toda") mostra a divisão da base inteira, para comparação."""
    tabela = pd.crosstab(dados[grupo], dados[col], normalize='index')
    tabela.index = [rotulo_grupo(g) for g in tabela.index]
    tabela.loc['Base toda'] = dados[col].value_counts(normalize=True)
    tabela = tabela.iloc[::-1]   # barh desenha de baixo para cima: assim a primeira categoria fica no topo

    inicio = np.zeros(len(tabela))
    for categoria, cor in zip(tabela.columns, cores):
        valores = tabela[categoria].values
        nome = nomes[categoria] if nomes else categoria
        barras = ax.barh(tabela.index, valores, left=inicio, color=cor, edgecolor='white', linewidth=2, label=nome)
        ax.bar_label(barras, labels=[f'{v:.1%}' if v >= 0.06 else '' for v in valores], label_type='center',
                     fontsize=9, color='#0b0b0b' if cor in CORES_TEXTO_ESCURO else 'white')
        inicio += valores

    ax.axhline(0.5, color=COR_REFERENCIA, linewidth=1)   # separa a "Base toda" dos grupos
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis='y', visible=False)
    ax.set_ylabel('')
    ax.set_title(f'Variável {col}')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
