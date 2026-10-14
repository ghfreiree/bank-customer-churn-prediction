"""Busca de hiperparâmetros com validação cruzada aninhada, usada para todos os modelos do notebook 02_modelagem."""
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_validate

from src.config import RANDOM_STATE


def contar_treinos(params, cv):
    """Mostra quantas combinações de hiperparâmetros existem e quantos treinos o loop interno faria com todas.
    `params` pode ser um dicionário ou uma lista de dicionários (como na Logistic Regression)."""
    grades = params if isinstance(params, list) else [params]
    total = sum(np.prod([len(v) for v in grade.values()]) for grade in grades)
    print(f'{total} combinações x {cv.get_n_splits()} dobras = {total * cv.get_n_splits()} treinos')


def mostrar_hiperparametros(busca):     # Hiperparâmetros escolhidos pela busca (smote__ e nearmiss__ ficam, para saber de qual passo são)
    print('Melhores hiperparâmetros:')
    for k, v in busca.best_params_.items():
        print(f'  {k.replace("modelo__", ""):<20} = {v}')


def avaliar_busca_aninhada(pipe, params, x, y, scoring, inner_cv, outer_cv, n_iter=40):
    """Validação cruzada aninhada:
    - loop interno (`inner_cv`): o RandomizedSearchCV escolhe os hiperparâmetros pelo F2;
    - loop externo (`outer_cv`): estima o desempenho honesto do processo de busca.
    Depois refaz a busca com todo o treino, para ter o modelo final.

    Retorna a busca já treinada e os scores das dobras externas."""
    busca = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=n_iter,
        scoring=scoring,
        refit='f2',
        cv=inner_cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=True,   # permite comparar treino x validação no cv_results_
    )

    scores = cross_validate(busca, x, y, cv=outer_cv, scoring=scoring, n_jobs=-1)
    busca.fit(x, y)

    print(f'F2 em cada dobra externa: {scores["test_f2"].round(3)}')
    print(f'Média dos F2: {scores["test_f2"].mean():.3f}\n')
    mostrar_hiperparametros(busca)

    return busca, scores
