from functools import partial
from itertools import groupby
from multiprocessing import Pool
from typing import List, Tuple, Union
import tqdm
import numpy as np
from sklearn.metrics import average_precision_score


def f(args, y_true1, y_pred1, y_true2, y_pred2):
    idx, weights, old1, old2, seq_id = args
    weights_arr = np.array(weights)[None, None, :]
    new1 = average_precision_score(y_true1, (y_pred1[:, :, idx] * weights_arr).sum(-1))
    new2 = average_precision_score(y_true2, (y_pred2[:, :, idx] * weights_arr).sum(-1))
    return idx, weights, new1, new2, old1, old2, seq_id


def beam_search(
        y_true1: np.ndarray,
        y_pred1: np.ndarray,
        y_true2: np.ndarray,
        y_pred2: np.ndarray,
        window: int = 8,
        weights: Union[List[int], Tuple[int]] = (0.3, 0.6, 1.0),
        n_jobs: int = 10,
        verbose: bool = True
):
    """
    Отбор моделей в ансамбль бим сёрчем по двум фолдам.
    * условие добавления модели - после добавления фичи скор должен увеличиться на обоих фолдах: в таком случае разница
    между фолдами, как правило, получается меньше, чем если просто максимизировать среднее
    * для каждой модели можно подобрать свой вес. веса-кандидаты прокидываются через параметр weights.
    * если модель добавляется в ансамль, то она добавляется с одним лучшим весом. сделал так, чтоб увеличить разнообразие
    ансамблей-кандидатов.
    * запуск мультипроцессинга в первом цикле вместо второго сильно ускоряет алгоритм

    N - число объектов
    C - число классов
    M - число моделей-кандидатов в ансамбль
    y_true - [N, C]
    y_pred - [N, C, M]
    """
    best = [([], [], -1, -1, False)]  # храним w пар (последовательность, веса, скор, is_terminated)
    weights = weights if weights is not None else [1]
    g = partial(
        f, y_true1=y_true1, y_pred1=y_pred1, y_true2=y_true2, y_pred2=y_pred2
    )
    nm = y_pred1.shape[-1]  # num models
    with tqdm.tqdm() as pbar:
        while not all(x[4] for x in best):
            curr = []
            args = []
            # seq_id - для группировки родительских последовательностей
            for seq_id, (seq, w, s1, s2, is_terminated) in enumerate(best):
                # сохраняем закончившуюся последовательность чтоб не забыть о ней
                if is_terminated:
                    curr.append((seq, w, s1, s2, True))
                    continue

                # генерим кандидатов
                new_feats = set(range(nm)) - set(seq)
                for i in new_feats:
                    # в случае пустой последоватлеьности добавляем каждую фичу с весом 1
                    weights_i = weights if len(seq) > 0 else [1]
                    for j in weights_i:
                        args.append((seq + [i], w + [j], s1, s2, seq_id))

            # удаляем дубликаты
            id2x = {frozenset(zip(x[0], x[1])): x for x in args}
            args = list(id2x.values())

            # скорим кандидатов
            with Pool(n_jobs) as p:
                tmp = p.map(g, args)

            # группируем по последовательности
            for _, g in groupby(sorted(tmp, key=lambda x: x[-1]), key=lambda x: x[-1]):

                # группируем по фиче
                # добавляем фичу с лучшим весом
                best_weighted = []
                for _, h in groupby(sorted(g, key=lambda x: x[0][-1]), key=lambda x: x[0][-1]):
                    y = max(h, key=lambda x: x[2] + x[3])
                    best_weighted.append(y)

                # ищем кандидатов, у которых скор улучшился на обоих фолдах
                is_terminated = True
                x = None
                for x in best_weighted:
                    if x[2] > x[4] and x[3] > x[5]:
                        curr.append((x[0], x[1], x[2], x[3], False))
                        is_terminated = False

                # если не удалось найти ни одного кандидата, то записываем в curr родителя, чтоб не забыть о нём
                if is_terminated:
                    assert x is not None
                    # нужны старые фичи, веса и скоры. для этого подойдёт любой элемент группы
                    curr.append((x[0][:-1], x[1][:-1], x[4], x[5], True))

            # берём top-window кандидатов
            best = sorted(curr, key=lambda x: -(x[2] + x[3]))[:window]
            if verbose:
                print(best)
                print("=" * 50)
            pbar.update(1)
    return best
