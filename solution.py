#!/usr/bin/env python
# ДЗ 3. Обучение ранжированию — шаблон решения.
#
# Обучение:  ./solution.py --train --model_file=model.bin PATH-TO-DATA
# Сабмишн:   ./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA

import argparse
import catboost
import os
from timeit import default_timer as timer
from catboost import utils

import pandas as pd


def train(data_dir, model_file):
    data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    data = data.sort_values('QueryId', kind='stable').reset_index(drop=True)

    X = data.drop('Label', axis = 1)
    y = data['Label']

    train_idx = data['QueryId'] < 8000
    test_idx = data['QueryId'] >= 8000

    X_train = X[train_idx].drop(['QueryId'], axis = 1)
    y_train = y[train_idx]
    X_test = X[test_idx].drop(['QueryId'], axis = 1)
    y_test = y[test_idx]

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    q_train = data.loc[train_idx, 'QueryId'].to_numpy()
    q_test = data.loc[test_idx, 'QueryId'].to_numpy()

    pool_train = catboost.Pool(X_train, label=y_train, group_id=q_train)
    pool_test = catboost.Pool(X_test, label=y_test, group_id=q_test)
    PARAMS = {
        "loss_function": "YetiRank",
        "iterations": 2500,
        "eval_metric": "NDCG:top=10;type=Exp",
        "random_seed": 42,
        "task_type": "CPU",
        "thread_count": -1,
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3,
        'early_stopping_rounds': 400
    }

    model = catboost.CatBoost(PARAMS)
    model.fit(
        pool_train,
        eval_set=pool_test,
        use_best_model=True,
        verbose=200
    )
    model.save_model(model_file)

    y_pred = model.predict(X_test)
    scores = utils.eval_metric(
        y_test,
        y_pred,
        'NDCG:top=10;type=Exp',
        group_id=q_test
    )
    print(f"Финальный скор nDCG@10 = {scores[0]}")


def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    X = test.drop(['QueryId', 'ObjectId'], axis = 1)

    model = catboost.CatBoost()
    model.load_model(model_file)
    scores = model.predict(X)

    submission = pd.DataFrame({'ObjectId': test['ObjectId'], 'Score': scores})
    submission.to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Learning to rank homework solution')
    parser.add_argument('--train', action='store_true',
                        help='обучить модель и сохранить её в model_file')
    parser.add_argument('--model_file', required=True, help='файл модели')
    parser.add_argument('--submission_file',
                        help='куда записать сабмишн для Kaggle')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()

    # Если в решении есть случайность, зафиксируйте seed: random_seed у CatBoost, seed у LightGBM и т. п.

    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error('без --train нужен --submission_file')
        predict(args.data_dir, args.model_file, args.submission_file)

    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
