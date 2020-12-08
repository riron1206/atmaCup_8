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

# https://www.kaggle.com/yxohrxn/votingclassifier-fit

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
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

from sklearn.model_selection import KFold, GroupKFold

warnings.simplefilter("ignore")
pd.set_option("display.max_columns", None)

# +
from sklearn.metrics import mean_squared_log_error


def score(y, y_pred):
    RMSLE = np.sqrt(np.mean(((np.log(y + 1) - np.log(y_pred + 1)) ** 2)))
    # RMSLE = mean_squared_log_error(y, y_pred) ** 0.5
    return RMSLE


# +
import numpy as np


def objective_for_weights(weights, Y_true, Y_preds):
    Y_pred = np.tensordot(weights, Y_preds, axes=(0, 0))

    return score(Y_true, Y_pred)


# + code_folding=[3]
import numpy as np


class ObjectiveWithEarlyStopping(object):
    def __init__(
        self, y_true, y_preds, y_true_valid, y_preds_valid, patience=30
    ):
        self.y_true = np.asarray(y_true)
        self.y_preds = np.asarray(y_preds)
        self.y_true_valid = np.asarray(y_true_valid)
        self.y_preds_valid = np.asarray(y_preds_valid)
        self.patience = patience

        self._nit = 0
        self._wait = 0
        self._best_score = np.inf
        self._best_weights = None

    def __call__(self, params):
        train_score = self._objective(
            params,
            self.y_true,
            self.y_preds,
        )
        valid_score = self._objective(
            params,
            self.y_true_valid,
            self.y_preds_valid,
        )

        self._nit += 1

        if valid_score < self._best_score:
            self._wait = 0
            self._best_score = valid_score
            self._best_params = params
        else:
            self._wait += 1

        if self._wait >= self.patience:
            raise RuntimeError(f"Epoch {self._nit}: early stopping")

        return train_score
    

class ObjectiveForWeightsWithEarlyStopping(ObjectiveWithEarlyStopping):
    @property
    def _objective(self):
         return objective_for_weights


# -

# # Data

DATADIR = r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\notebook\lgbm\20201206_Publisher_GroupkFold_seed"
train = pd.read_csv(f"{DATADIR}/train_fe.csv")
test = pd.read_csv(f"{DATADIR}/test_fe.csv")

# +
Y_preds = []
paths = []

_DATADIR = r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\notebook\lgbm\20201206_Publisher_GroupkFold"
path = f"{_DATADIR}/Y_pred_psude_label.pkl"
with open(path, "rb") as f:
    y_pred = pickle.load(f)
y_pred = y_pred.iloc[: train.shape[0], :]  # 複数クラス試す
#y_pred = y_pred.iloc[: train.shape[0], [0]]  # 1クラスだけ試す
columns = y_pred.columns
y_pred = np.asarray(y_pred)
Y_preds.append(y_pred)
paths.append(path)
display(y_pred)
    
_DATADIR = r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\notebook\xgb\20201206_Publisher_GroupkFold"
path = f"{_DATADIR}/Y_pred_psude_label.pkl"
with open(path, "rb") as f:
    y_pred = pickle.load(f)
y_pred = y_pred.iloc[: train.shape[0], :]  # 複数クラス試す
#y_pred = y_pred.iloc[: train.shape[0], [0]]  # 1クラスだけ試す
columns = y_pred.columns
y_pred = np.asarray(y_pred)
Y_preds.append(y_pred)
paths.append(path)
display(y_pred)

Y_preds = np.asarray(Y_preds)
print("Y_preds.shape:", Y_preds.shape)

n_models = len(Y_preds)

result = pd.DataFrame(index=paths)

print("columns:", columns)

target_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]  # 複数クラス試す
#target_cols = ["NA_Sales"]  # 1クラスだけ試す

# +
sales_cols = [
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales",
]

train_drop_sales = train.drop(sales_cols, axis=1)

# +
# group_col = "Name"
group_col = "Publisher"  # https://www.guruguru.science/competitions/13/discussions/42fc473d-4450-4cfc-b924-0a5d61fd0ca7/

# GroupKFold
group = train_drop_sales[group_col].copy()  

# seed値が指定できるGroupKFold  https://www.guruguru.science/competitions/13/discussions/cc7167cb-3627-448a-b9eb-7afcd29fd122/
group = train_drop_sales[group_col].copy()
group_uni = train_drop_sales[group_col].copy().unique()

# +
# test setとあまり被りのない特徴量は除外  https://www.guruguru.science/competitions/13/discussions/df06ef19-981d-4666-a0c0-22f62ee26640/

inbalance = ["Publisher", "Developer", "Name"]
train_drop_sales = train_drop_sales.drop(inbalance, axis=1)
test = test.drop(inbalance, axis=1)

cate_cols = ['Platform', 'Rating', 'Year_of_Release', 'series', 'Genre']
print("cate_cols:", cate_cols)

# +
features = test.columns.to_list()

global_target_col = "Global_Sales"

X = train_drop_sales[features].copy()
Y = train[target_cols].copy()
Y_global = train[global_target_col].copy()

train_size, n_features = X.shape

_, n_classes = Y.shape
# -

n_seeds = 5
n_splits = 5
shuffle = True

# +
corr = np.empty((n_models, n_models))
corr = pd.DataFrame(corr, columns=paths, index=paths)

for i, row in enumerate(paths):
    for j, column in enumerate(paths):
        if i <= j:
            corr.loc[row, column] = 0
        else:
            df = pd.DataFrame(Y_preds[i])
            other = pd.DataFrame(Y_preds[j])
            corr.loc[row, column] = df.corrwith(other).mean()

corr.style.background_gradient(cmap="Blues", subset=paths, vmax=1.0, vmin=0.0)
# -

# # ObjectiveForWeightsWithEarlyStopping

# +
Y_pred = np.zeros((train_size, n_classes))
Y_pred = pd.DataFrame(Y_pred, columns=Y.columns, dtype="float", index=Y.index)

weights = np.zeros((n_classes, n_models))
n_iters = np.zeros(n_classes)

x0 = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {
    "type": "eq",
    "fun": lambda x: np.sum(x) - 1.0,
    "jac": lambda x: np.ones_like(x),
}
options = {"ftol": 0.0, "maxiter": 1_000_000}

for i in range(n_seeds):
    cv = KFold(n_splits=n_splits, random_state=i, shuffle=shuffle)
    cv_split = cv.split(group_uni)

    for j, (trn_idx, val_idx) in enumerate(cv_split):
        for k, column in enumerate(columns):
            objective = ObjectiveForWeightsWithEarlyStopping(
                Y.iloc[trn_idx, [k]],
                Y_preds[:, trn_idx, [k]],
                Y.iloc[val_idx, [k]],
                Y_preds[:, val_idx, [k]],
                patience=30,
            )

            try:
                res = minimize(
                    objective,
                    x0,
                    bounds=bounds,
                    constraints=constraints,
                    method="SLSQP",
                    options=options,
                )
            except RuntimeError:
                pass

            weights[k] += objective._best_params / n_seeds / n_splits
            n_iters[k] += objective._nit / n_seeds / n_splits
            
            Y_pred.iloc[val_idx, k] += np.tensordot(
                objective._best_params, Y_preds[:, val_idx, [k]], axes=(0, 0)
            ) / n_seeds

with open("weights.pkl", "wb") as f:
    pickle.dump(weights, f)
# -

weights

# +
result["weights_mean"] = np.mean(weights, axis=0)

result.style.background_gradient(cmap="Blues")

# +
result = pd.DataFrame(weights, columns=paths, index=Y.columns)
result["n_iter"] = n_iters

result.style.background_gradient(cmap="Blues", subset=paths, vmax=1.0, vmin=0.0)
# +
for i, column in enumerate(columns):
    Y_pred[column] = np.tensordot(weights[i], Y_preds[:, :, [i]], axes=(0, 0))
    
score(Y, Y_pred)
# -


score(Y.sum(axis=1), Y_pred.sum(axis=1))

Y

Y_pred


