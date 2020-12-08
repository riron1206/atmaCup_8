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

from catboost import CatBoost, Pool

warnings.simplefilter("ignore")
pd.set_option("display.max_columns", None)

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:96% !important; }</style>"))  # デフォルトは75%

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


# + code_folding=[7]
import seaborn as sns
import matplotlib.pyplot as plt


def display_importances(
    importance_df, png_path=f"feature_importance.png",
):
    """feature_importance plot"""
    # importance_df.sort_values(by="importance", ascending=False).to_csv(
    #    f"feature_importance.csv"
    # )
    cols = (
        importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:100]
        .index
    )
    best_features = importance_df.loc[importance_df.feature.isin(cols)]
    plt.figure(figsize=(8, 15))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False),
    )
    plt.title(f"CatBoost (avg over folds)   {png_path}")
    plt.tight_layout()
    plt.savefig(png_path)


# + code_folding=[]
import category_encoders as ce


def count_encoder(
    X_train, Y_train, X_val, Y_val, target_col: str, cat_features=None, features=None
):
    """
    Count_Encoding: カテゴリ列をカウント値に変換する特徴量エンジニアリング（要はgroupby().size()の集計列追加のこと）
    ※カウント数が同じカテゴリは同じようなデータ傾向になる可能性がある
    https://www.kaggle.com/matleonard/categorical-encodings
    """
    X_train = pd.DataFrame(X_train, columns=features)
    Y_train = pd.DataFrame(Y_train, columns=[target_col])
    X_val = pd.DataFrame(X_val, columns=features)
    Y_val = pd.DataFrame(Y_val, columns=[target_col])

    train_df = X_train.join(Y_train)
    valid_df = X_val.join(Y_val)

    count_enc = ce.CountEncoder(cols=cat_features)

    # trainだけでfitすること(validationやtest含めるとリークする)
    count_enc.fit(train_df[cat_features])
    train_encoded = train_df.join(
        count_enc.transform(train_df[cat_features]).add_suffix("_count")
    )
    valid_encoded = valid_df.join(
        count_enc.transform(valid_df[cat_features]).add_suffix("_count")
    )

    features = train_encoded.drop(target_col, axis=1).columns.to_list()

    # return train_encoded, valid_encoded
    return (
        train_encoded.drop(target_col, axis=1),
        valid_encoded.drop(target_col, axis=1),
        features,
    )


def target_encoder(
    X_train, Y_train, X_val, Y_val, target_col: str, cat_features=None, features=None
):
    """
    Target_Encoding: カテゴリ列を目的変数の平均値に変換する特徴量エンジニアリング
    https://www.kaggle.com/matleonard/categorical-encodings
    """
    X_train = pd.DataFrame(X_train, columns=features)
    Y_train = pd.DataFrame(Y_train, columns=[target_col])
    X_val = pd.DataFrame(X_val, columns=features)
    Y_val = pd.DataFrame(Y_val, columns=[target_col])

    train_df = X_train.join(Y_train)
    valid_df = X_val.join(Y_val)

    target_enc = ce.TargetEncoder(cols=cat_features)

    # trainだけでfitすること(validationやtest含めるとリークする)
    target_enc.fit(train_df[cat_features], train_df[target_col])

    train_encoded = train_df.join(
        target_enc.transform(train_df[cat_features]).add_suffix("_target")
    )
    valid_encoded = valid_df.join(
        target_enc.transform(valid_df[cat_features]).add_suffix("_target")
    )

    features = train_encoded.drop(target_col, axis=1).columns.to_list()

    # return train_encoded, valid_encoded
    return (
        train_encoded.drop(target_col, axis=1),
        valid_encoded.drop(target_col, axis=1),
        features,
    )


def catboost_encoder(
    X_train, Y_train, X_val, Y_val, target_col: str, cat_features=None, features=None
):
    """
    CatBoost_Encoding: カテゴリ列を目的変数の1行前の行からのみに変換する特徴量エンジニアリング
    CatBoost使ったターゲットエンコーディング
    https://www.kaggle.com/matleonard/categorical-encodings
    """
    X_train = pd.DataFrame(X_train, columns=features)
    Y_train = pd.DataFrame(Y_train, columns=[target_col])
    X_val = pd.DataFrame(X_val, columns=features)
    Y_val = pd.DataFrame(Y_val, columns=[target_col])

    train_df = X_train.join(Y_train)
    valid_df = X_val.join(Y_val)

    cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=7)

    # trainだけでfitすること(validationやtest含めるとリークする)
    cb_enc.fit(train_df[cat_features], train_df[target_col])

    train_encoded = train_df.join(
        cb_enc.transform(train_df[cat_features]).add_suffix("_cb")
    )
    valid_encoded = valid_df.join(
        cb_enc.transform(valid_df[cat_features]).add_suffix("_cb")
    )

    features = train_encoded.drop(target_col, axis=1).columns.to_list()

    # return train_encoded, valid_encoded
    return (
        train_encoded.drop(target_col, axis=1),
        valid_encoded.drop(target_col, axis=1),
        features,
    )


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
# -

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

# # FE

df = pd.concat([train_drop_sales, test], axis=0)


# +
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
    # df_all[f"row_{col_name}_skew"] = df.skew(axis=1)  # 歪度 Nan になるからやめる

    return df_all


df = add_num_row_agg(df, ["Critic_Score", "User_Score"])
df = add_num_row_agg(df, ["Critic_Count", "User_Count"])

# + code_folding=[6, 24]
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
        "std",  # 標準偏差
        "skew",  # 歪度
        pd.DataFrame.kurt,  # 尖度
    ],
    "Critic_Count": [
        "max",
        "min",
        "mean",
        "std",  # 標準偏差
        "skew",  # 歪度
        pd.DataFrame.kurt,  # 尖度
    ],
    "User_Score": [
        "max",
        "min",
        "mean",
        "std",  # 標準偏差
        "skew",  # 歪度
        pd.DataFrame.kurt,  # 尖度
    ],
    "Critic_Score": [
        "max",
        "min",
        "mean",
        "std",  # 標準偏差
        "skew",  # 歪度
        pd.DataFrame.kurt,  # 尖度
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
# KMeansでクラスタリングした列追加

from sklearn.cluster import KMeans


def fe_cluster(df, kind, features, n_clusters=100, SEED=42, is_dummies=False):
    df_ = df[features].copy()
    kmeans_cells = KMeans(n_clusters=n_clusters, random_state=SEED).fit(df_)
    df[f"clusters_{kind}"] = kmeans_cells.predict(df_.values)
    df = pd.get_dummies(df, columns=[f"clusters_{kind}"]) if is_dummies else df

    return df


df = fe_cluster(
    df,
    kind="cate_cols",
    features=["Name", "Genre", "Publisher", "Developer", "Rating"],
    n_clusters=3000,
)
df = fe_cluster(
    df,
    kind="Score_Count",
    features=["Critic_Score", "User_Score", "Critic_Count", "User_Count"],
    n_clusters=300,
)

# +
train_drop_sales = df.iloc[: train.shape[0]]
test = df.iloc[train.shape[0] :].reset_index(drop=True)

pd.concat([train_drop_sales, train[sales_cols]], axis=1).to_csv("train_fe.csv", index=False)
test.to_csv("test_fe.csv", index=False)
# -

train_drop_sales

test

# # params

# +
n_seeds = 1
n_splits = 5
shuffle = True

params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
#    "depth": 12,
#    "random_strength": 0.1,
#    "l2_leaf_reg": 10,
#    "subsample": 0.9,
    "num_boost_round": 20000,
    "task_type": "GPU",
}
verbose_eval = 100
early_stopping_rounds = verbose_eval

# catbostに渡すカテゴリ型のカラム
cat_features = cate_cols  # + ["clusters_cate_cols", "clusters_Score_Count"]

#DEBUG = True
DEBUG = False
if DEBUG:
    n_seeds = 1
    n_splits = 2
    params["learning_rate"] = 0.1
    params["num_boost_round"] = 100
    verbose_eval = 50
    early_stopping_rounds = verbose_eval

# +
features = test.columns.to_list()

global_target_col = "Global_Sales"
target_cols = [
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
]

X = train_drop_sales[features].copy()
Y = train[target_cols].copy()
Y_global = train[global_target_col].copy()

train_size, n_features = X.shape
# group = X["Name"].copy()
group = X[
    "Publisher"
].copy()  # https://www.guruguru.science/competitions/13/discussions/42fc473d-4450-4cfc-b924-0a5d61fd0ca7/
_, n_classes = Y.shape
# -

# # catboost

# +
# %%time

f_importance = {col: None for col in Y.columns}
Y_pred = np.zeros((train_size, n_classes))
Y_pred = pd.DataFrame(Y_pred, columns=Y.columns, index=Y.index)

for i in tqdm(range(n_seeds)):
    set_seed(seed=i)
    params["random_state"] = i

    # cv = KFold(n_splits=n_splits, random_state=i, shuffle=shuffle)
    # cv_split = cv.split(X, Y)
    cv = GroupKFold(n_splits=n_splits)
    cv_split = cv.split(X, Y, group)

    for j, (trn_idx, val_idx) in enumerate(cv_split):
        print(f"\n------------ fold:{j} ------------")

        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        Y_train_targets, Y_val_targets = Y.iloc[trn_idx], Y.iloc[val_idx]

        for tar, tar_col in enumerate(Y.columns):
            print(f"\n------------ target:{tar_col} ------------")
            Y_train, Y_val = (
                Y_train_targets.values[:, tar],
                Y_val_targets.values[:, tar],
            )

            X_train_enc, X_val_enc, features_enc = (
                X_train.copy(),
                X_val.copy(),
                features.copy(),
            )
            # count_encoder はやめとく。存在しないNameやPublisherなnanaになるので
            # X_train_enc, X_val_enc, features_enc = count_encoder(X_train_enc, Y_train, X_val_enc, Y_val, target_col=tar_col, cat_features=cate_cols, features=features_enc)
            # X_train_enc, X_val_enc, features_enc = target_encoder(X_train_enc, Y_train, X_val_enc, Y_val, target_col=tar_col, cat_features=cate_cols, features=features_enc,)
            # X_train_enc, X_val_enc, features_enc = catboost_encoder(X_train_enc, Y_train, X_val_enc, Y_val, target_col=tar_col, cat_features=cate_cols, features=features_enc)

            # display(X_val_enc.head())

            #train_pool = Pool(X_train, Y_train, cat_features=cat_features)
            #eval_pool = Pool(X_val, Y_val, cat_features=cat_features)
            train_pool = Pool(X_train_enc, label=Y_train, cat_features=cat_features)
            eval_pool = Pool(X_val_enc, label=Y_val, cat_features=cat_features)

            model = CatBoost(params)
            model.fit(train_pool, eval_set=eval_pool, 
                      use_best_model=True, 
                      early_stopping_rounds=early_stopping_rounds,
                      verbose_eval=verbose_eval,
                     )
            
            # Y_pred[tar_col][val_idx] += model.predict(X_val) / n_seeds
            Y_pred[tar_col][val_idx] += model.predict(X_val_enc) / n_seeds

            imp = np.array(model.get_feature_importance()) / (n_seeds * n_splits)
            if f_importance[tar_col] is None:
                f_importance[tar_col] = imp
            else:
                f_importance[tar_col] += imp

            joblib.dump(model, f"model_seed_{i}_fold_{j}_{tar_col}.jlb", compress=True)

for col in Y.columns:
    idx = Y_pred[Y_pred[col] < 0].index
    Y_pred[col][idx] = 0.0

with open("Y_pred.pkl", "wb") as f:
    pickle.dump(Y_pred, f)

# -
score(Y, Y_pred)

score(Y_global, Y_pred.sum(axis=1))


# +
def _importance(features, n_features):
    df_f_imps = np.zeros((n_features, len(f_importance)))
    df_f_imps = pd.DataFrame(df_f_imps, columns=Y.columns, index=model.feature_names_)

    for _, tar_col in enumerate(Y.columns):
        df_importance = pd.DataFrame(
            {"feature": model.feature_names_, "importance": f_importance[tar_col]}
        )
        display_importances(df_importance, png_path=f"{tar_col}_feature_importance")

        df_f_imps[tar_col] = f_importance[tar_col]

    df_f_imps.to_csv("feature_importance.csv")
    return df_f_imps


# _importance(features, n_features)
df_f_imps = _importance(features_enc, len(features_enc))
df_f_imps
# -

# # oof

# +
# path = r"Y_pred.pkl"
# with open(path, "rb") as f:
#    Y_pred = pickle.load(f)
# Y_pred
# -

# # predict test

# +
X_test = test.copy()
Y_test_pred = pd.Series(np.zeros((test.shape[0])))

for i in range(n_seeds):
    for j in range(n_splits):
        for tar_col in Y.columns:
            model_path = f"model_seed_{i}_fold_{j}_{tar_col}.jlb"
            model = joblib.load(model_path)

            X_enc, X_test_enc, features_enc = X.copy(), X_test.copy(), features.copy()
            # count_encoder はやめとく。存在しないNameやPublisherなnanaになるので
            # X_enc, X_test_enc, features_enc = count_encoder(X_enc, Y[tar_col], X_test_enc, Y_test_pred, target_col=tar_col, cat_features=cate_cols, features=features_enc)
            # X_enc, X_test_enc, features_enc = target_encoder(X_enc, Y[tar_col], X_test_enc, Y_test_pred, target_col=tar_col, cat_features=cate_cols, features=features_enc,)
            # X_enc, X_test_enc, features_enc = catboost_encoder(X_enc, Y[tar_col], X_test_enc, Y_test_pred, target_col=tar_col, cat_features=cate_cols, features=features_enc)

            # display(X_test_enc.head())

            # _y_pred = model.predict(X_test) / (n_seeds * n_splits)
            _y_pred = model.predict(X_test_enc) / (n_seeds * n_splits)
            _y_pred[_y_pred < 0] = 0.0

            Y_test_pred += _y_pred

print(Y_test_pred.shape)

submission = pd.read_csv(f"{DATADIR}/atmaCup8_sample-submission.csv")
submission["Global_Sales"] = Y_test_pred
submission.to_csv("submission.csv", index=False)
submission
# -

# # psude_label

# + code_folding=[]
# 疑似ラベル作成
X_test = test.copy()
Y_test_pred = np.zeros((test.shape[0], n_classes))
Y_test_pred = pd.DataFrame(Y_test_pred, columns=target_cols, index=test.index)

for i in range(n_seeds):
    for j in range(n_splits):
        for tar_col in Y.columns:
            model_path = f"model_seed_{i}_fold_{j}_{tar_col}.jlb"
            model = joblib.load(model_path)

            X_enc, X_test_enc, features_enc = X.copy(), X_test.copy(), features.copy()
            # count_encoder はやめとく。存在しないNameやPublisherなnanaになるので
            # X_enc, X_test_enc, features_enc = count_encoder(X_enc, Y[tar_col], X_test_enc, Y_test_pred, target_col=tar_col, cat_features=cate_cols, features=features_enc)
            # X_enc, X_test_enc, features_enc = target_encoder(X_enc, Y[tar_col], X_test_enc, Y_test_pred, target_col=tar_col, cat_features=cate_cols, features=features_enc,)
            # X_enc, X_test_enc, features_enc = catboost_encoder(X_enc, Y[tar_col], X_test_enc, Y_test_pred, target_col=tar_col, cat_features=cate_cols, features=features_enc)

            # display(X_test_enc.head())

            # _y_pred = model.predict(X_test) / (n_seeds * n_splits)
            _y_pred = model.predict(X_test_enc) / (n_seeds * n_splits)
            _y_pred[_y_pred < 0] = 0.0

            Y_test_pred[tar_col] += _y_pred
Y_test_pred

# +
# 疑似ラベル連結
X_test = test.copy()

X = pd.concat([X, X_test], axis=0).reset_index(drop=True)
Y = pd.concat([Y, Y_test_pred], axis=0).reset_index(drop=True)

train_size, n_features = X.shape
# group = X["Name"].copy()
group = X[
    "Publisher"
].copy()  # https://www.guruguru.science/competitions/13/discussions/42fc473d-4450-4cfc-b924-0a5d61fd0ca7/
_, n_classes = Y.shape

print(X.shape, Y.shape)

# +
# importance 上位100位までの特徴量だけにする
features_imp = sorted(
    pd.DataFrame(df_f_imps.sum(axis=1))
    .sort_values(by=0, ascending=False)[:120]
    .index.to_list()
)

features_imp

# +
# %%time

# 疑似ラベル付けて再学習

Y_pred = np.zeros((train_size, n_classes))
Y_pred = pd.DataFrame(Y_pred, columns=Y.columns, index=Y.index)

for i in tqdm(range(n_seeds)):
    set_seed(seed=i)
    params["random_state"] = i

    # cv = KFold(n_splits=n_splits, random_state=i, shuffle=shuffle)
    # cv_split = cv.split(X, Y)
    cv = GroupKFold(n_splits=n_splits)
    cv_split = cv.split(X, Y, group)

    for j, (trn_idx, val_idx) in enumerate(cv_split):
        print(f"\n------------ fold:{j} ------------")

        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        Y_train_targets, Y_val_targets = Y.iloc[trn_idx], Y.iloc[val_idx]

        for tar, tar_col in enumerate(Y.columns):
            print(f"\n------------ target:{tar_col} ------------")
            Y_train, Y_val = (
                Y_train_targets.values[:, tar],
                Y_val_targets.values[:, tar],
            )

            X_train_enc, X_val_enc, features_enc = (
                X_train.copy(),
                X_val.copy(),
                features.copy(),
            )
            # count_encoder はやめとく。存在しないNameやPublisherなnanaになるので
            # X_train_enc, X_val_enc, features_enc = count_encoder(X_train_enc, Y_train, X_val_enc, Y_val, target_col=tar_col, cat_features=cate_cols, features=features_enc)
            # X_train_enc, X_val_enc, features_enc = target_encoder(X_train_enc, Y_train, X_val_enc, Y_val, target_col=tar_col, cat_features=cate_cols, features=features_enc,)
            # X_train_enc, X_val_enc, features_enc = catboost_encoder(X_train_enc, Y_train, X_val_enc, Y_val, target_col=tar_col, cat_features=cate_cols, features=features_enc)

            # importance 上位100位までの特徴量だけにする
            X_train_enc, X_val_enc = X_train_enc[features_imp], X_val_enc[features_imp]

            # display(X_val_enc.head())

            #train_pool = Pool(X_train, Y_train, cat_features=cat_features)
            #eval_pool = Pool(X_val, Y_val, cat_features=cat_features)
            train_pool = Pool(X_train_enc, label=Y_train, cat_features=cat_features)
            eval_pool = Pool(X_val_enc, label=Y_val, cat_features=cat_features)

            model = CatBoost(params)
            model.fit(train_pool, eval_set=eval_pool, 
                      use_best_model=True, 
                      early_stopping_rounds=early_stopping_rounds,
                      verbose_eval=verbose_eval,
                     )
            
            # Y_pred[tar_col][val_idx] += model.predict(X_val) / n_seeds
            Y_pred[tar_col][val_idx] += model.predict(X_val_enc) / n_seeds
            
            joblib.dump(model, f"model_seed_{i}_fold_{j}_{tar_col}.jlb", compress=True)

for col in Y.columns:
    idx = Y_pred[Y_pred[col] < 0].index
    Y_pred[col][idx] = 0.0
    
with open("Y_pred_psude_label.pkl", "wb") as f:
    pickle.dump(Y_pred.iloc[: train.shape[0]], f)
    
# -

score(Y.iloc[: train.shape[0]], Y_pred.iloc[: train.shape[0]])

score(Y_global, Y_pred.iloc[: train.shape[0]].sum(axis=1))

# +
# predict test

X_test = test.copy()
Y_test_pred = pd.Series(np.zeros((test.shape[0])))

for i in range(n_seeds):
    for j in range(n_splits):
        for tar_col in Y.columns:
            model_path = f"model_seed_{i}_fold_{j}_{tar_col}.jlb"
            model = joblib.load(model_path)

            X_enc, X_test_enc, features_enc = X.copy(), X_test.copy(), features.copy()
            # count_encoder はやめとく。存在しないNameやPublisherなnanaになるので
            # X_enc, X_test_enc, features_enc = count_encoder(X_enc, Y[tar_col], X_test_enc, Y_test_pred, target_col=tar_col, cat_features=cate_cols, features=features_enc)
            X_enc, X_test_enc, features_enc = target_encoder(X_enc, Y[tar_col], X_test_enc, Y_test_pred, target_col=tar_col, cat_features=cate_cols, features=features_enc,)
            # X_enc, X_test_enc, features_enc = catboost_encoder(X_enc, Y[tar_col], X_test_enc, Y_test_pred, target_col=tar_col, cat_features=cate_cols, features=features_enc)

            # importance 上位100位までの特徴量だけにする
            X_test_enc = X_test_enc[features_imp]

            # display(X_test_enc.head())

            # _y_pred = model.predict(X_test) / (n_seeds * n_splits)
            _y_pred = model.predict(X_test_enc) / (n_seeds * n_splits)
            _y_pred[_y_pred < 0] = 0.0

            Y_test_pred += _y_pred

print(Y_test_pred.shape)

submission = pd.read_csv(f"{DATADIR}/atmaCup8_sample-submission.csv")
submission["Global_Sales"] = Y_test_pred
submission.to_csv("submission_psude_label.csv", index=False)
submission
# -

