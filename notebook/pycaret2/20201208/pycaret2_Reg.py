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

# + [markdown] papermill={"duration": 0.092568, "end_time": "2020-08-07T07:17:20.608866", "exception": false, "start_time": "2020-08-07T07:17:20.516298", "status": "completed"} tags=[]
# # PyCaret 2 Regression Example
# - https://github.com/pycaret/pycaret/blob/master/examples/PyCaret%202%20Regression.ipynb

# + execution={"iopub.execute_input": "2020-08-07T07:17:20.800167Z", "iopub.status.busy": "2020-08-07T07:17:20.799149Z", "iopub.status.idle": "2020-08-07T07:17:20.827968Z", "shell.execute_reply": "2020-08-07T07:17:20.828775Z"} papermill={"duration": 0.127113, "end_time": "2020-08-07T07:17:20.829021", "exception": false, "start_time": "2020-08-07T07:17:20.701908", "status": "completed"} tags=[]
from pycaret.utils import version
version()

# + execution={"iopub.execute_input": "2020-08-07T07:17:21.220348Z", "iopub.status.busy": "2020-08-07T07:17:21.219385Z", "iopub.status.idle": "2020-08-07T07:17:28.141603Z", "shell.execute_reply": "2020-08-07T07:17:28.140751Z"} papermill={"duration": 7.02273, "end_time": "2020-08-07T07:17:28.141797", "exception": false, "start_time": "2020-08-07T07:17:21.119067", "status": "completed"} tags=[]
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:96% !important; }</style>"))# デフォルトは75%

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm_notebook as tqdm
import pycaret

sns.set()

# 常に全ての列（カラム）を表示
pd.set_option("display.max_columns", None)

# !pwd
sys.executable

# + execution={"iopub.execute_input": "2020-08-07T07:17:28.365378Z", "iopub.status.busy": "2020-08-07T07:17:28.364121Z", "iopub.status.idle": "2020-08-07T07:17:28.423886Z", "shell.execute_reply": "2020-08-07T07:17:28.422742Z"} papermill={"duration": 0.185137, "end_time": "2020-08-07T07:17:28.424215", "exception": false, "start_time": "2020-08-07T07:17:28.239078", "status": "completed"} tags=["parameters"]
# パラメ
# https://pycaret.readthedocs.io/en/latest/api/regression.html

# 出力ディレクトリ
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 目的変数
target = "Global_Sales"

# 乱数シード
session_id = 123

# cv hold
fold = 5
fold_strategy = "groupkfold"
fold_groups = "Publisher"

# metric
optimize = "RMSLE"

# models
choice_ms = ["et", "catboost", "lightgbm", "gbr", "huber"]

# チューニング回数
n_iter = 100

#DEBUG = True
DEBUG = False
if DEBUG:
    n_iter = 2

params = dict(
    target=target,
    session_id=session_id,
    fold=fold,
    fold_strategy=fold_strategy,
    fold_groups=fold_groups,
    fold_shuffle=True,
    silent=True,
    use_gpu=True,
)

# +
import pandas as pd

#DATADIR = r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\data\atmacup08-dataset"
#train = pd.read_csv(f"{DATADIR}/train.csv")
#test = pd.read_csv(f"{DATADIR}/test.csv")

#DATADIR = r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\notebook\data_check"
#train = pd.read_csv(f"{DATADIR}/train_preprocess.csv")
#test = pd.read_csv(f"{DATADIR}/test_preprocess.csv")

DATADIR = r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\notebook\lgbm\20201207_Publisher_GroupkFold_seed"
train = pd.read_csv(f"{DATADIR}/train_fe.csv")
test = pd.read_csv(f"{DATADIR}/test_fe.csv")

train = train.loc[:, test.columns.to_list() + ["Global_Sales"]]
train
# -

from pycaret.regression import *
reg1 = setup(train, **params)

best_model = compare_models(fold=fold,
                            groups=fold_groups,
                            sort=optimize)

tune_models = []
for name in choice_ms:
    print("model:", name)
    model = create_model(name, 
                         fold=fold, 
                         groups=fold_groups)
    tuned_m = tune_model(
        model,
        fold=fold,
        groups=fold_groups,
        optimize=optimize,
        n_iter=n_iter,
        early_stopping=True,
        early_stopping_max_iters=n_iter//3,
    )
    tune_models.append(tuned_m)
    save_model(tuned_m, model_name=os.path.join(output_dir, "pycaret_tuned_" + name))
print(tune_models)

# +
# groupkfold ではなぜかエラー
#stacker = stack_models(estimator_list=tune_models[:-1], 
#                       meta_model=tune_models[-1],
#                       fold=fold,
#                       groups=fold_groups,
#                       optimize=optimize,
#                      )
#save_model(stacker, model_name=os.path.join(output_dir, "pycaret_stacker"))

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
evaluate_model(tune_models[0])

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
f_tune_models = []
for name in choice_ms:
    loaded_model = load_model(os.path.join(output_dir, f"pycaret_tuned_{name}"))
    f_tune_models.append(loaded_model)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[] code_folding=[]
from sklearn.metrics import mean_squared_log_error

def score(y, y_pred):
    RMSLE = np.sqrt(np.mean(((np.log(y + 1) - np.log(y_pred + 1)) ** 2)))
    return RMSLE

def make_submit(df_test, model, output_dir: str, csv_name: str):
    """submit csv作成"""
    df_sub = predict_model(model, data=df_test)[["Label"]].reset_index()
    df_sub.columns = ["id", target]
    df_sub = df_sub[[target]]
    
    df_sub[target] = np.where(df_sub[target]<0, 0, df_sub[target])
    
    df_sub.to_csv(f"{output_dir}/{csv_name}.csv", index=False)
    return df_sub


for m, name in zip(f_tune_models, choice_ms):
    print(f"name: {name}")
    df_sub = make_submit(train.loc[:, test.columns.to_list()], m, output_dir, f"{name}_train_rmsle")
    y = train[target].values
    y_pred = df_sub[target].values
    y_pred = np.where(y_pred<0, 0, y_pred)
    print("train_rmsle:", score(y, y_pred))

for m, name in zip(f_tune_models, choice_ms):
    df_sub = make_submit(test, m, output_dir, f"{name}_submission")
    display(df_sub.head())
# -




