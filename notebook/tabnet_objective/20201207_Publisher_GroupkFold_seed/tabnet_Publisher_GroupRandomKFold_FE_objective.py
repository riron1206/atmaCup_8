# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # TF

# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# +
import tensorflow as tf
import tensorflow.keras.backend as K

import sys

sys.path.append(
    r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\notebook\tabnet"
)
from tabnet_tf import TabNetRegressor, StackedTabNetRegressor

# -

from adabelief_tf import AdaBeliefOptimizer

# +
import tensorflow as tf


def build_callbacks(
    model_path, factor=0.1, mode="auto", monitor="val_loss", patience=0, verbose=0
):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        mode=mode, monitor=monitor, patience=patience, verbose=verbose
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, mode=mode, monitor=monitor, save_best_only=True, verbose=verbose
    )
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        factor=factor, monitor=monitor, mode=mode, verbose=verbose
    )

    return [early_stopping, model_checkpoint, reduce_lr_on_plateau]


# -

# # base

# +
import os
import gc
import re
import math
import pickle
import joblib
import warnings

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from sklearn.model_selection import KFold, GroupKFold

import lightgbm as lgb

warnings.simplefilter("ignore")
pd.set_option("display.max_columns", None)

# + code_folding=[4]
import random as rn
import numpy as np


def set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)

    rn.seed(seed)
    np.random.seed(seed)


# + code_folding=[]
from sklearn.metrics import mean_squared_log_error


def score(y, y_pred):
    RMSLE = np.sqrt(np.mean(((np.log(y + 1) - np.log(y_pred + 1)) ** 2)))
    # RMSLE = mean_squared_log_error(y, y_pred) ** 0.5
    return RMSLE


# -

# # Data load

# +
import pandas as pd

DATADIR = r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\data\atmacup08-dataset"
train = pd.read_csv(f"{DATADIR}/train.csv")
test = pd.read_csv(f"{DATADIR}/test.csv")
df = pd.concat([train, test], axis=0)
# -

# # 前処理

# +
import numpy as np

# tbd(確認中)を欠損にする
df["User_Score"] = df["User_Score"].replace("tbd", np.nan)

# + code_folding=[1]
# -1で補完
cate_cols = [
    "Name",
    "Platform",
    "Year_of_Release",
    "Genre",
    "Publisher",
    "Developer",
    "Rating",
]
for col in cate_cols:
    df[col].fillna(-1, inplace=True)


# + code_folding=[2]
def impute_null_add_flag_col(
    df, strategy="mean", cols_with_missing=None, fill_value=None
):
    """欠損値を補間して欠損フラグ列を追加する
    fill_value はstrategy="constant"の時のみ有効になる補間する定数
    """
    from sklearn.impute import SimpleImputer

    df_plus = df.copy()

    for col in cols_with_missing:
        # 欠損フラグ列を追加
        df_plus[col + "_was_missing"] = df[col].isnull()
        df_plus[col + "_was_missing"] = df_plus[col + "_was_missing"].astype(int)
        # 欠損値を平均値で補間
        my_imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        df_plus[col] = my_imputer.fit_transform(df[[col]])

    return df_plus


df = impute_null_add_flag_col(
    df, strategy="most_frequent", cols_with_missing=["User_Score"]
)  # 最頻値で補間

df = impute_null_add_flag_col(
    df, cols_with_missing=["Critic_Score", "Critic_Count", "User_Count",],
)  # 平均値で補間（数値列のみ）

# +
# User_Scoreを数値列にする
df["User_Score"] = df["User_Score"].astype("float")

# User_Scoreを文字列にする
df["Year_of_Release"] = df["Year_of_Release"].astype("str")
# -

# ラベルエンコディング
cate_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
for col in cate_cols:
    df[col], uni = pd.factorize(df[col])

# +
train = df.iloc[: train.shape[0]]
test = df.iloc[train.shape[0] :].reset_index(drop=True)

# 目的変数
sales_cols = [
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales",
]

train_drop_sales = train.drop(sales_cols, axis=1)
test = test.drop(sales_cols, axis=1)
# -

# 欠損データ確認
_df = pd.DataFrame({"is_null": df.isnull().sum()})
_df[_df["is_null"] > 0]

# # FE

df = pd.concat([train_drop_sales, test], axis=0)


# + code_folding=[1]
# 行単位で統計量とる
def add_num_row_agg(df_all, agg_num_cols):
    """行単位の統計量列追加
    agg_num_cols は数値列だけでないとエラー"""
    import warnings

    warnings.filterwarnings("ignore")

    df = df_all[agg_num_cols]
    cols = df.columns.to_list()
    cols = map(str, cols)  # 文字列にする
    col_name = "_".join(cols)

    df_all[f"row_{col_name}_sum"] = df.sum(axis=1)
    df_all[f"row_{col_name}_mean"] = df.mean(axis=1)
    df_all[f"row_{col_name}_std"] = df.std(axis=1)
    # df_all[f"row_{col_name}_skew"] = df.skew(axis=1)  # 歪度  # nanになるから入れない

    return df_all


df = add_num_row_agg(df, ["Critic_Score", "User_Score"])
df = add_num_row_agg(df, ["Critic_Count", "User_Count"])

# + code_folding=[6, 23]
import numpy as np
import pandas as pd


# A列でグループして集計したB列は意味がありそうと仮説たててから統計値列作ること
# 目的変数をキーにして集計するとリークしたターゲットエンコーディングになるため説明変数同士で行うこと
def grouping(df, cols, agg_dict, prefix=""):
    """特定のカラムについてgroup化された特徴量の作成を行う
    Args:
        df (pd.DataFrame): 特徴量作成のもととなるdataframe
        cols (str or list): group by処理のkeyとなるカラム (listで複数指定可能)
        agg_dict (dict): 特徴量作成を行いたいカラム/集計方法を指定するdictionary
        prefix (str): 集約後のカラムに付与するprefix name

    Returns:
        df (pd.DataFrame): 特定のカラムについてgroup化された特徴量群
    """
    group_df = df.groupby(cols).agg(agg_dict)
    group_df.columns = [prefix + c[0] + "_" + c[1] for c in list(group_df.columns)]
    group_df.reset_index(inplace=True)

    return group_df


class AggUtil:
    ############## カテゴリ列 vs. 数値列について ##############
    @staticmethod
    def percentile(n):
        """パーセンタイル"""

        def percentile_(x):
            return np.percentile(x, n)

        percentile_.__name__ = "percentile_%s" % n
        return percentile_

    @staticmethod
    def diff_percentile(n1, n2):
        """パーセンタイルの差"""

        def diff_percentile_(x):
            p1 = np.percentile(x, n1)
            p2 = np.percentile(x, n2)
            return p1 - p2

        diff_percentile_.__name__ = f"diff_percentile_{n1}-{n2}"
        return diff_percentile_

    @staticmethod
    def ratio_percentile(n1, n2):
        """パーセンタイルの比"""

        def ratio_percentile_(x):
            p1 = np.percentile(x, n1)
            p2 = np.percentile(x, n2)
            return p1 / p2

        ratio_percentile_.__name__ = f"ratio_percentile_{n1}-{n2}"
        return ratio_percentile_

    @staticmethod
    def mean_var():
        """平均分散"""

        def mean_var_(x):
            x = x.dropna()
            return np.std(x) / np.mean(x)

        mean_var_.__name__ = f"mean_var"
        return mean_var_

    @staticmethod
    def diff_mean():
        """平均との差の中央値(aggは集計値でないとエラーになるから中央値をとる)"""

        def diff_mean_(x):
            x = x.dropna()
            return np.median(x - np.mean(x))

        diff_mean_.__name__ = f"diff_mean"
        return diff_mean_

    @staticmethod
    def ratio_mean():
        """平均との比の中央値(aggは一意な値でないとエラーになるから中央値をとる)"""

        def ratio_mean_(x):
            x = x.dropna()
            return np.median(x / np.mean(x))

        ratio_mean_.__name__ = f"ratio_mean"
        return ratio_mean_

    @staticmethod
    def hl_ratio():
        """平均より高いサンプル数と低いサンプル数の比率"""

        def hl_ratio_(x):
            x = x.dropna()
            n_high = x[x >= np.mean(x)].shape[0]
            n_low = x[x < np.mean(x)].shape[0]
            if n_low == 0:
                return 1.0
            else:
                return n_high / n_low

        hl_ratio_.__name__ = f"hl_ratio"
        return hl_ratio_

    @staticmethod
    def ratio_range():
        """最大/最小"""

        def ratio_range_(x):
            x = x.dropna()
            if np.min(x) == 0:
                return 1.0
            else:
                return np.max(x) / np.min(x)

        ratio_range_.__name__ = f"ratio_range"
        return ratio_range_

    @staticmethod
    def beyond1std():
        """1stdを超える比率"""

        def beyond1std_(x):
            x = x.dropna()
            return x[np.abs(x) > np.abs(np.std(x))].shape[0] / x.shape[0]

        beyond1std_.__name__ = "beyond1std"
        return beyond1std_

    @staticmethod
    def zscore():
        """Zスコアの中央値(aggは一意な値でないとエラーになるから中央値をとる)"""

        def zscore_(x):
            x = x.dropna()
            return np.median((x - np.mean(x)) / np.std(x))

        zscore_.__name__ = "zscore"
        return zscore_

    ######################################################

    ############## カテゴリ列 vs. カテゴリ列について ##############
    @staticmethod
    def freq_entropy():
        """出現頻度のエントロピー"""
        from scipy.stats import entropy

        def freq_entropy_(x):
            return entropy(x.value_counts().values)

        freq_entropy_.__name__ = "freq_entropy"
        return freq_entropy_

    @staticmethod
    def freq1name():
        """最も頻繁に出現するカテゴリの数"""

        def freq1name_(x):
            return x.value_counts().sort_values(ascending=False)[0]

        freq1name_.__name__ = "freq1name"
        return freq1name_

    @staticmethod
    def freq1ratio():
        """最も頻繁に出現するカテゴリ/グループの数"""

        def freq1ratio_(x):
            frq = x.value_counts().sort_values(ascending=False)
            return frq[0] / frq.shape[0]

        freq1ratio_.__name__ = "freq1ratio"
        return freq1ratio_

    #########################################################


# 集計する数値列指定
value_agg = {
    "User_Count": [
        "max",
        "min",
        "mean",
        # "std",  # 標準偏差
        # "skew",  # 歪度
        # pd.DataFrame.kurt,  # 尖度
    ],
    "Critic_Count": [
        "max",
        "min",
        "mean",
        # "std",  # 標準偏差
        # "skew",  # 歪度
        # pd.DataFrame.kurt,  # 尖度
    ],
    "User_Score": [
        "max",
        "min",
        "mean",
        # "std",  # 標準偏差
        # "skew",  # 歪度
        # pd.DataFrame.kurt,  # 尖度
    ],
    "Critic_Score": [
        "max",
        "min",
        "mean",
        # "std",  # 標準偏差
        # "skew",  # 歪度
        # pd.DataFrame.kurt,  # 尖度
    ],
}
# グループ化するカテゴリ列でループ
for key in ["Platform", "Genre", "Publisher", "Developer", "Rating"]:
    feature_df = grouping(df, key, value_agg, prefix=key + "_")
    df = pd.merge(df, feature_df, how="left", on=key)
# -

df["Critic_Score_*_Critic_Count"] = df["Critic_Score"] * df["Critic_Count"]
df["User_Score_*_User_Count"] = df["User_Score"] * df["User_Count"]
df["Critic_Score_*_User_Score"] = df["Critic_Score"] * df["User_Score"]
df["Critic_Count_*_User_Count"] = df["Critic_Count"] * df["User_Count"]
df["Critic_Count_+_User_Count"] = df["Critic_Count"] + df["User_Count"]
df["Critic_Count_-_User_Count"] = df["Critic_Count"] - df["User_Count"]
df["Critic_Count_/_all_Count"] = df["Critic_Count"] / df["Critic_Count_+_User_Count"]

# + code_folding=[]
## KMeansでクラスタリングした列追加
#
# from sklearn.cluster import KMeans
#
# def fe_cluster(df, kind, features, n_clusters=100, SEED=42, is_dummies=False):
#    df_ = df[features].copy()
#    kmeans_cells = KMeans(n_clusters=n_clusters, random_state=SEED).fit(df_)
#    df[f'clusters_{kind}'] = kmeans_cells.predict(df_.values)
#    df = pd.get_dummies(df, columns=[f'clusters_{kind}']) if is_dummies else df
#
#    return df
#
# df = fe_cluster(df, kind="cate_cols",
#                features=["Name", "Genre", "Publisher", "Developer", "Rating"],
#                n_clusters=3000)
# df = fe_cluster(df, kind="Score_Count",
#                features=["Critic_Score", "User_Score", "Critic_Count", "User_Count"],
#                n_clusters=300)

# +
# 種類数少な目の列はダミー化して残す

float_cols = df.columns.to_list()

for col in [
    "Critic_Count_was_missing",
    "Critic_Score_was_missing",
    "User_Count_was_missing",
    "User_Score_was_missing",
]:
    float_cols.remove(col)

df = pd.get_dummies(df, columns=[f"Rating"])
df = pd.get_dummies(df, columns=[f"Genre"])
df = pd.get_dummies(df, columns=[f"Platform"])

# 種類数少な目の列は消しとく
df = df.drop(["Name", "Year_of_Release", "Publisher", "Developer"], axis=1)

float_cols = np.intersect1d(np.array(float_cols), np.array(df.columns.to_list()))
float_cols

# +
## RankGauss
# from sklearn.preprocessing import QuantileTransformer
#
# _df = df[float_cols]
#
# qt = QuantileTransformer(n_quantiles=100, random_state=42, output_distribution="normal")
# df[_df.columns] = qt.fit_transform(_df)
# -

# 欠損データ確認
_df = pd.DataFrame({"is_null": df.isnull().sum()})
_df[_df["is_null"] > 0]

train_drop_sales = df.iloc[: train.shape[0]]
test = df.iloc[train.shape[0] :].reset_index(drop=True)

train_drop_sales

test

# # tabnet

# +
n_seeds = 1
n_splits = 5
shuffle = True

# batch_size = 64
batch_size = 1024
factor = 0.5
patience = 30
lr = 0.001
# fit_params = {"epochs": 1_000, "verbose": 1}
# fit_params = {"epochs": 300, "verbose": 1}
fit_params = {"epochs": 30, "verbose": 0}

# DEBUG = True
DEBUG = False
if DEBUG:
    n_seeds = 1
    n_splits = 2
    batch_size = 1024
    fit_params = {"epochs": 10, "verbose": 1}
    print("DEBUG")

# +
# _train = pd.read_csv(f"{DATADIR}/train.csv")

# group_col = "Name"
group_col = "Publisher"  # https://www.guruguru.science/competitions/13/discussions/42fc473d-4450-4cfc-b924-0a5d61fd0ca7/

# GroupKFold
group = train[group_col].copy()

# seed値が指定できるGroupKFold  https://www.guruguru.science/competitions/13/discussions/cc7167cb-3627-448a-b9eb-7afcd29fd122/
group_uni = train[group_col].copy().unique()

# +
features = test.columns.to_list()

global_target_col = "Global_Sales"
target_cols = [
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
]

X = train_drop_sales[features]
Y = train[target_cols]
Y_global = train[global_target_col]

train_size, n_features = X.shape
_, n_classes = Y.shape


# + code_folding=[0, 5]
def root_mean_squared_logarithmic_error(y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    return K.sqrt(msle(y_true, y_pred))


def fit_tabnet(params, X_train, Y_train, X_val, Y_val, model_path):
    K.clear_session()
    model = TabNetRegressor(num_regressors=n_classes, num_features=n_features, **params)
    model.compile(
        optimizer=AdaBeliefOptimizer(learning_rate=lr),
        # loss=tf.keras.losses.MeanSquaredLogarithmicError(),  # MSLE
        loss=root_mean_squared_logarithmic_error,  # RMSLE
    )

    callbacks = build_callbacks(model_path, factor=factor, patience=patience)

    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=(X_val, Y_val),
        **fit_params,
    )

    model.load_weights(model_path)

    return model


# + code_folding=[]
def fit_model(params):

    Y_pred = np.zeros((train_size, n_classes))
    Y_pred = pd.DataFrame(Y_pred, columns=Y.columns, index=Y.index)

    for i in tqdm(range(n_seeds)):
        set_seed(seed=i)

        cv = KFold(n_splits=n_splits, random_state=i, shuffle=shuffle)
        # cv_split = cv.split(X, Y)  # KFold
        cv_split = cv.split(group_uni)  # GroupRandomKFold
        # cv = GroupKFold(n_splits=n_splits)
        # cv_split = cv.split(X, Y, group)

        for j, (trn_idx, val_idx) in enumerate(cv_split):
            print(f"\n------------ fold:{j} ------------")
            # GroupRandomKFold
            tr_groups, va_groups = group_uni[trn_idx], group_uni[val_idx]
            trn_idx = group[group.isin(tr_groups)].index
            val_idx = group[group.isin(va_groups)].index
            print("len(trn_idx), len(val_idx):", len(trn_idx), len(val_idx))

            X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            Y_train, Y_val = Y.iloc[trn_idx], Y.iloc[val_idx]

            model = fit_tabnet(
                params, X_train, Y_train, X_val, Y_val, f"model_seed_{i}_fold_{j}.h5"
            )

            va_pred = model.predict(X_val)

            va_pred = np.where(va_pred < 0, 0, va_pred)

            Y_pred.iloc[val_idx] += va_pred / n_seeds

            del model
            gc.collect()

    return score(Y_global, Y_pred.sum(axis=1))


# -

# # objective

# +
import optuna


def objective(trial):
    params = dict(
        epsilon=1e-05,
        feature_columns=None,
        virtual_batch_size=None,
        num_groups=-1,
        batch_momentum=0.9,
        relaxation_factor=1.2,
        sparsity_coefficient=0.0001,
        norm_type=trial.suggest_categorical("norm_type", ["group", "batch"]),  # 正規化のタイプ
        num_decision_steps=trial.suggest_categorical(
            "Nsteps", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ),
    )
    params["feature_dim"] = trial.suggest_categorical(
        "Na", [8, 16, 24, 32, 64, 128, 512, 1024]
    )
    params["output_dim"] = int(
        trial.suggest_uniform(
            "Nd", int(params["feature_dim"] // 8), int(params["feature_dim"] // 2)
        )
    )
    print("params:", params)

    return fit_model(params)


# +
import warnings

warnings.filterwarnings("ignore")

n_trials = 300
if DEBUG:
    n_trials = 3

study = optuna.create_study(
    study_name="study",
    storage=f"sqlite:///study.db",
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=1),
)
study.optimize(objective, n_trials=n_trials)
study.trials_dataframe().to_csv(f"objective_history.csv", index=False)
with open(f"objective_best_params.txt", mode="w") as f:
    f.write(str(study.best_params))
print(f"\nstudy.best_params:\n{study.best_params}")
print(f"\nstudy.best_value:\n{study.best_value}")
# -
