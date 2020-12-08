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

import lightgbm as lgb

warnings.simplefilter("ignore")

# +
import random as rn
import numpy as np


def set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)

    rn.seed(seed)
    np.random.seed(seed)


# +
from sklearn.metrics import mean_squared_log_error


def score(y, y_pred):
    RMSLE = np.sqrt(np.mean(((np.log(y + 1) - np.log(y_pred + 1)) ** 2)))
    #RMSLE = mean_squared_log_error(y, y_pred) ** 0.5
    return RMSLE


# +
import seaborn as sns
import matplotlib.pyplot as plt


def display_importances(
    importance_df,
    png_path=f"feature_importance.png",
):
    """feature_importance plot"""
    importance_df.sort_values(by="importance", ascending=False).to_csv(
        f"feature_importance.csv"
    )
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
    plt.title(f"LightGBM (avg over folds)   {png_path}")
    plt.tight_layout()
    plt.savefig(png_path)


# -

# # Data

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
    df,
    cols_with_missing=[
        "Critic_Score",
        "Critic_Count",
        "User_Count",
    ],
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

test = test.drop(sales_cols, axis=1)
# -

test

train

# # lgbm

# +
n_seeds = 5
n_splits = 5
shuffle = True

params = {
    "objective": "root_mean_squared_error",
    "learning_rate": 0.01,
    "feature_fraction": 0.5,
    "lambda_l1": 8.647619848158392,
    "lambda_l2": 1.0881077642285376e-09,
    "max_depth": 7,
    "min_data_in_leaf": 1,
    "num_leaves": 111,
    "reg_sqrt": True,
}

num_boost_round = 20000
verbose_eval = 100
early_stopping_rounds = verbose_eval


# DEBUG = True
DEBUG = False
if DEBUG:
    params["learning_rate"] = 0.1
    n_seeds = 2
    n_splits = 2
    num_boost_round = 10
    verbose_eval = 5
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

X = train[features]
Y = train[target_cols]
Y_global = train[global_target_col]

train_size, n_features = X.shape
group = X["Name"]
_, n_classes = Y.shape

# +
# %%time

f_importance = {col: np.zeros((n_features)) for col in Y.columns} 
Y_pred = np.zeros((train_size, n_classes))
Y_pred = pd.DataFrame(Y_pred, columns=Y.columns, index=Y.index)

for i in tqdm(range(n_seeds)):
    set_seed(seed=i)

    cv = KFold(n_splits=n_splits, random_state=i, shuffle=shuffle)
    cv_split = cv.split(X, Y)
    #cv = GroupKFold(n_splits=n_splits)
    #cv_split = cv.split(X, Y, group)

    for j, (trn_idx, val_idx) in enumerate(cv_split):
        print(f"\n------------ fold:{j} ------------")

        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        Y_train_targets, Y_val_targets = Y.iloc[trn_idx], Y.iloc[val_idx]
    
        for tar, tar_col in enumerate(Y.columns):
            Y_train, Y_val = Y_train_targets.values[:, tar], Y_val_targets.values[:, tar]  

            lgb_train = lgb.Dataset(X_train, Y_train)
            lgb_eval = lgb.Dataset(X_val, Y_val, reference=lgb_train)

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=verbose_eval,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
            Y_pred[tar_col][val_idx] += (
                model.predict(X_val, num_iteration=model.best_iteration) / n_seeds
            )

            f_importance[tar_col] += np.array(model.feature_importance(importance_type="gain")) / (n_seeds * n_splits)

            joblib.dump(model, f"model_seed_{i}_fold_{j}_{tar_col}.jlb", compress=True)

for col in Y.columns:
    idx = Y_pred[Y_pred[col] < 0].index
    Y_pred[col][idx] = 0.0
            
with open("Y_pred.pkl", "wb") as f:
    pickle.dump(Y_pred, f)
# -
score(Y, Y_pred)

score(Y_global, Y_pred.sum(axis=1))

for _, tar_col in enumerate(Y.columns):
    df_importance = pd.DataFrame({"feature": model.feature_name(), "importance": f_importance[tar_col]})
    display_importances(df_importance, png_path=f"{tar_col}_feature_importance")

# # oof

path = r"Y_pred.pkl"
with open(path, "rb") as f:
    Y_pred = pickle.load(f)
Y_pred

# # predict test

# +
Y_test_pred = np.zeros((test.shape[0]))

for i in range(n_seeds):
    for j in range(n_splits):
        for tar_col in Y.columns:
            model_path = f"model_seed_{i}_fold_{j}_{tar_col}.jlb"
            model = joblib.load(model_path)

            _y_pred = model.predict(test) / (n_seeds * n_splits)
            _y_pred[_y_pred < 0] = 0.0
            
            Y_test_pred += _y_pred
            
print(Y_test_pred.shape)

submission = pd.read_csv(f"{DATADIR}/atmaCup8_sample-submission.csv")
submission["Global_Sales"] = Y_test_pred
submission.to_csv("submission.csv", index=False)
submission
# -


