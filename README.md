# ДЗ 3. Обучение ранжированию

## Результат на Kaggle

- **Ник на Kaggle:** `coinlord`
- **Скор на публичном лидерборде (nDCG@10):** `0.62352`

## Решение

Для ранжирования используется CatBoost с функцией потерь `YetiRank` на исходных числовых признаках. `QueryId` задаёт группы документов и не используется как признак. `Label` — целевая переменная, `ObjectId` используется только для формирования сабмита.

Выборка делится по запросам: `QueryId < 8000` — обучение, `QueryId >= 8000` — валидация. Документы одного запроса целиком попадают в одну часть. Метрика — `NDCG:top=10;type=Exp`.

Лучшая итерация выбирается по валидации с помощью early stopping. Сохраняется модель, обученная на обучающей части; переобучение на всём `train.csv` в текущем скрипте не выполняется.

## Лучшие гиперпараметры

Конфигурация, используемая в `solution.py`:

| Параметр | Значение |
|---|---|
| `loss_function` | `YetiRank` |
| `iterations` | `2500` |
| `depth` | `5` |
| `learning_rate` | `0.03` |
| `l2_leaf_reg` | `3` |
| `early_stopping_rounds` | `400` |
| `eval_metric` | `NDCG:top=10;type=Exp` |
| `random_seed` | `42` |
| `task_type` | `CPU` |
| `thread_count` | `-1` |
| `use_best_model` (при обучении) | `True` |

`iterations` задаёт максимальное число деревьев; итоговое число определяется по валидации.

## Запуск

Установить зависимости:

```bash
python -m pip install -r requirements.txt
```

Положить `train.csv` и `test.csv` в папку `data/`.

Обучить модель:

```bash
python solution.py --train --model_file=model.bin data
```

Сформировать сабмит:

```bash
python solution.py --model_file=model.bin --submission_file=submission.csv data
```

Результат — файл `submission.csv` с колонками `ObjectId,Score`, по одной строке для каждого объекта из `test.csv`.

## Зависимости

- `catboost==1.2.10`
- `pandas==2.2.2`

Версии зафиксированы в `requirements.txt`.
