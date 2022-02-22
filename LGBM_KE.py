#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt


class LearningVisualizationCallback(object):
    """学習の過程を動的にプロットするコールバック"""

    def __init__(self, fig=None, ax=None):
        self._metric_histories = defaultdict(list)
        self._metric_history_lines = {}
        self._metric_type_higher_better = {}
        self._best_score_lines = {}
        self._best_score_texts = {}

        # 初期化する
        self._fig = fig
        self._ax = ax
        if self._fig is None and self._ax is None:
            self._fig, self._ax = plt.subplots()
        self._ax.set_title('learning curve')
        self._ax.set_xlabel('round')
        self._ax.set_ylabel('score')
        self._fig.canvas.draw()
        self._fig.show()

    def __call__(self, env):
        # メトリックを保存する
        evals = env.evaluation_result_list
        for _, name, mean, is_higher_better, _ in evals:
            self._metric_histories[name].append(mean)

            # 初回だけの設定
            if env.iteration == 0:
                # メトリックの種別を保存する
                self._metric_type_higher_better[name] = is_higher_better
                # スコアの履歴を描画するオブジェクトを生成する
                history_line, = self._ax.plot([], [])
                history_line.set_label(name)
                self._metric_history_lines[name] = history_line
                # ベストスコアの線を描画するオブジェクトを生成する
                best_line = self._ax.axhline(0)
                best_line.set_color(history_line.get_color())
                best_line.set_linestyle(':')
                self._best_score_lines[name] = best_line
                # ベストスコアの文字列を描画するオブジェクトを生成する
                best_text = self._ax.text(0, 0, '', weight='bold')
                best_text.set_color(history_line.get_color())
                self._best_score_texts[name] = best_text

        # 可視化する
        for name, values in self._metric_histories.items():
            # グラフデータを更新する
            history_line = self._metric_history_lines[name]
            history_line.set_data(np.arange(len(values)), values)
            best_line = self._best_score_lines[name]
            best_find_func = np.max if self._metric_type_higher_better[name] else np.min
            best_score = best_find_func(values)
            best_line.set_ydata(best_score)
            best_text = self._best_score_texts[name]
            best_text.set_text('{:.6f}'.format(best_score))
            best_text.set_y(best_score)

        # グラフの見栄えを調整する
        self._ax.legend()
        self._ax.relim()
        self._ax.autoscale_view()

        # 再描画する
        plt.pause(0.001)

    def show_until_close(self):
        """ウィンドウを閉じるまで表示し続ける"""
        plt.show()


def accuracy(preds, data):
    """精度 (Accuracy) を計算する関数
    NOTE: 表示が eval set の LogLoss だけだと寂しいので"""
    y_true = data.get_label()
    y_pred = np.where(preds > 0.5, 1, 0)
    acc = np.mean(y_true == y_pred)
    return 'accuracy', acc, True


def main():
    # Titanic データセットを読み込む
    #dataset = sns.load_dataset('titanic')

    # 重複など不要な特徴量は落とす
    # X = dataset.drop(['survived',
    #                  'class',
    #                  'who',
    #                  'embark_town',
    #                  'alive'], axis=1)
    #y = dataset.survived


    # カテゴリカル変数を指定する
    #categorical_columns = ['pclass',
    #                       'sex',
    #                       'embarked',
    #                       'adult_male',
    #                       'deck',
    #                       'alone']
    #X = X.astype({c: 'category'
    #              for c in categorical_columns})

    df = pd.read_csv('data1719_dfr.csv')
    X = df.drop(['Trait'], axis=1) # 説明変数のみにする
    y = df['Trait']  # 正解クラス


    # LightGBM のデータセット表現に直す
    lgb_train = lgb.Dataset(X, y)


    # 学習の過程を可視化するコールバックを用意する
    visualize_cb = LearningVisualizationCallback()
    callbacks = [
        visualize_cb,
    ]


    # 二値分類を LogLoss で評価する
    lgb_params = {
        'objective': 'binary',
        'metrics': 'binary_logloss',
    }


    # 5-Fold CV
    skf = StratifiedKFold(n_splits=5,
                          shuffle=True,
                          random_state=42)
    lgb.cv(lgb_params, lgb_train,
           num_boost_round=1000,
           early_stopping_rounds=100,
           verbose_eval=10,
           folds=skf, seed=42,
           # Accuracy も確認する
           feval=accuracy,
           # コールバックを登録する
           callbacks=callbacks)


    # ウィンドウを閉じるまで表示し続ける
    visualize_cb.show_until_close()


if __name__ == '__main__':
    main()