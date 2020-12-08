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
# check version
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
# 入力ディレクトリ
data_dir = r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\data\atmacup08-dataset"

# 出力ディレクトリ
output_dir = "20201204"
os.makedirs(output_dir, exist_ok=True)

input_csv = f"{data_dir}/eng2.csv"

# 目的変数
target = "Global_Sales"

# 乱数シード
session_id = 123

# cv hold
fold = 5

# metric
optimize = "Accuracy"

# models
choice_ms = ["gbc", "lightgbm", "catboost", ]
#choice_ms = ['nb', 'lightgbm', "rf"]  # test用

# チューニング回数
n_iter = 100
#n_iter = 3  # test用

# compare_models()時間かかるので
#is_compare_models = False
is_compare_models = True

# setup()のcsv保存するか（0.5GBぐらい容量食うから）
is_save_setup_csv = False
#is_save_setup_csv = True

# 学習に除く列
ignore_features = []

# +
import pandas as pd

#DATADIR = r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\data\atmacup08-dataset"
#train = pd.read_csv(f"{DATADIR}/train.csv")
#test = pd.read_csv(f"{DATADIR}/test.csv")

DATADIR = r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\notebook\data_check"
train = pd.read_csv(f"{DATADIR}/train_preprocess.csv")
test = pd.read_csv(f"{DATADIR}/test_preprocess.csv")

train = train.loc[:, test.columns.to_list() + ["Global_Sales"]]
train
# -

params = dict(target="Global_Sales", 
              session_id=123, 
              silent=True,
             )

# +
from pycaret.regression import *

reg1 = setup(train, **params)
# -

best_model = compare_models(fold=5, sort="RMSE")

lightgbm = create_model('lightgbm')

tuned_lightgbm = tune_model(lightgbm, n_iter=50, optimize = 'MAE')

tuned_lightgbm



dt = create_model('dt')

bagged_dt = ensemble_model(dt, n_estimators=50)

boosted_dt = ensemble_model(dt, method = 'Boosting')

blender = blend_models()

stacker = stack_models(
    estimator_list=compare_models(
        n_select=5, fold=5, whitelist=models(type="ensemble").index.tolist()
    )
)









# + [markdown] papermill={"duration": 0.122794, "end_time": "2020-08-07T07:17:28.916818", "exception": false, "start_time": "2020-08-07T07:17:28.794024", "status": "completed"} tags=[]
# # data load
# -

def preprocess(train_df, test_df, target_col):
    """前処理"""
    df = train_df.append(test_df).reset_index(drop=True)
    
    # label encoding. これしないとoehot encodingでメモリ死ぬ
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns.to_list():
        df[col], uni = pd.factorize(df[col])
    
    train_df = df[df[target_col].notnull()].reset_index(drop=True)
    train_df[target_col] = train_df[target_col].astype(int)
    test_df = df[df[target_col].isnull()].reset_index(drop=True)
    
    return train_df, test_df


# + execution={"iopub.execute_input": "2020-08-07T07:17:29.159159Z", "iopub.status.busy": "2020-08-07T07:17:29.158197Z", "iopub.status.idle": "2020-08-07T07:17:44.951633Z", "shell.execute_reply": "2020-08-07T07:17:44.952308Z"} papermill={"duration": 15.915309, "end_time": "2020-08-07T07:17:44.952599", "exception": false, "start_time": "2020-08-07T07:17:29.03729", "status": "completed"} tags=[]
df = pd.read_csv(input_csv, index_col=0)
df_train = df[df[target].notnull()].reset_index(drop=True)
df_test = df[df[target].isnull()].reset_index(drop=True)

df_train, df_test = preprocess(df_train, df_test, target)

print(df_train.info())
display(
    df_train.head().style.background_gradient(cmap="Pastel1")
)  
display(df_train.describe().style.background_gradient(cmap="Pastel1"))

print(df_test.info())
display(df_test.head().style.background_gradient(cmap="Pastel1"))
display(df_test.describe().style.background_gradient(cmap="Pastel1"))

# + execution={"iopub.execute_input": "2020-08-07T07:17:45.23002Z", "iopub.status.busy": "2020-08-07T07:17:45.229084Z", "iopub.status.idle": "2020-08-07T07:17:45.263882Z", "shell.execute_reply": "2020-08-07T07:17:45.262994Z"} papermill={"duration": 0.190496, "end_time": "2020-08-07T07:17:45.264053", "exception": false, "start_time": "2020-08-07T07:17:45.073557", "status": "completed"} tags=[]
params = {"target": target, 
          "session_id": session_id, 
          "silent": True,
          #"ignore_features": ignore_features,
         }

# + [markdown] papermill={"duration": 0.133949, "end_time": "2020-08-07T07:17:45.517494", "exception": false, "start_time": "2020-08-07T07:17:45.383545", "status": "completed"} tags=[]
# # 2. Initialize Setup

# + execution={"iopub.execute_input": "2020-08-07T07:17:45.794045Z", "iopub.status.busy": "2020-08-07T07:17:45.792676Z", "iopub.status.idle": "2020-08-07T07:17:45.833536Z", "shell.execute_reply": "2020-08-07T07:17:45.832574Z"} papermill={"duration": 0.17092, "end_time": "2020-08-07T07:17:45.833745", "exception": false, "start_time": "2020-08-07T07:17:45.662825", "status": "completed"} tags=[]
#from pycaret.classification import *
#help(setup)

# + code_folding=[] execution={"iopub.execute_input": "2020-08-07T07:17:46.093604Z", "iopub.status.busy": "2020-08-07T07:17:46.092265Z", "iopub.status.idle": "2020-08-07T07:20:14.897078Z", "shell.execute_reply": "2020-08-07T07:20:14.896002Z"} papermill={"duration": 148.942231, "end_time": "2020-08-07T07:20:14.89738", "exception": false, "start_time": "2020-08-07T07:17:45.955149", "status": "completed"} tags=[]
from pycaret.regression import *

if is_save_setup_csv:
    _df_test = df_test.copy()
    _df_test[target] = 0  # target列仮で入れる
    
    clf1 = setup(_df_test, **params)
    display(clf1[0].head(3))
    
    # 一応前処理後のtest set保存しておく
    pd.concat([clf1[0], clf1[1]], axis=1).to_csv(
        os.path.join(output_dir, "test_setup.csv"), index=False
    )  

clf1 = setup(df_train, **params)
display(clf1[0].head(3))

if is_save_setup_csv:
    # 一応前処理後のtrain set保存しておく
    pd.concat([clf1[0], clf1[1]], axis=1).to_csv(
        os.path.join(output_dir, "train_setup.csv"), index=False
    )  

# + execution={"iopub.execute_input": "2020-08-07T07:20:15.245921Z", "iopub.status.busy": "2020-08-07T07:20:15.244637Z", "iopub.status.idle": "2020-08-07T07:20:15.320133Z", "shell.execute_reply": "2020-08-07T07:20:15.318713Z"} papermill={"duration": 0.256342, "end_time": "2020-08-07T07:20:15.320508", "exception": false, "start_time": "2020-08-07T07:20:15.064166", "status": "completed"} tags=[]
# test
lr = create_model('lr', fold=fold)

# + [markdown] papermill={"duration": 0.178935, "end_time": "2020-08-07T07:20:15.674529", "exception": false, "start_time": "2020-08-07T07:20:15.495594", "status": "completed"} tags=[]
# # 3. Compare Baseline

# + execution={"iopub.execute_input": "2020-08-07T07:20:16.017324Z", "iopub.status.busy": "2020-08-07T07:20:16.015991Z", "iopub.status.idle": "2020-08-07T07:20:16.091768Z", "shell.execute_reply": "2020-08-07T07:20:16.090458Z"} papermill={"duration": 0.254372, "end_time": "2020-08-07T07:20:16.092054", "exception": false, "start_time": "2020-08-07T07:20:15.837682", "status": "completed"} tags=[]
if is_compare_models:
    best_model = compare_models(sort=optimize, fold=fold)

# + [markdown] papermill={"duration": 0.182413, "end_time": "2020-08-07T07:20:16.44788", "exception": false, "start_time": "2020-08-07T07:20:16.265467", "status": "completed"} tags=[]
# # 4. Create Model

# + execution={"iopub.execute_input": "2020-08-07T07:20:16.792665Z", "iopub.status.busy": "2020-08-07T07:20:16.791353Z", "iopub.status.idle": "2020-08-07T07:20:16.890649Z", "shell.execute_reply": "2020-08-07T07:20:16.891805Z"} papermill={"duration": 0.276918, "end_time": "2020-08-07T07:20:16.892118", "exception": false, "start_time": "2020-08-07T07:20:16.6152", "status": "completed"} tags=[]
models()

# + execution={"iopub.execute_input": "2020-08-07T07:20:17.258198Z", "iopub.status.busy": "2020-08-07T07:20:17.251826Z", "iopub.status.idle": "2020-08-07T07:20:17.343684Z", "shell.execute_reply": "2020-08-07T07:20:17.342463Z"} papermill={"duration": 0.274837, "end_time": "2020-08-07T07:20:17.344014", "exception": false, "start_time": "2020-08-07T07:20:17.069177", "status": "completed"} tags=[]
models(type="ensemble").index.tolist()

# + execution={"iopub.execute_input": "2020-08-07T07:20:17.702944Z", "iopub.status.busy": "2020-08-07T07:20:17.701324Z", "iopub.status.idle": "2020-08-07T07:20:17.776143Z", "shell.execute_reply": "2020-08-07T07:20:17.77506Z"} papermill={"duration": 0.255242, "end_time": "2020-08-07T07:20:17.776441", "exception": false, "start_time": "2020-08-07T07:20:17.521199", "status": "completed"} tags=[]
#ensembled_models = compare_models(
#    whitelist=models(type="ensemble").index.tolist(), fold=3
#)

# + [markdown] papermill={"duration": 0.171473, "end_time": "2020-08-07T07:20:18.127", "exception": false, "start_time": "2020-08-07T07:20:17.955527", "status": "completed"} tags=[]
# # 5. Tune Hyperparameters

# + execution={"iopub.execute_input": "2020-08-07T07:20:18.497303Z", "iopub.status.busy": "2020-08-07T07:20:18.495979Z"} papermill={"duration": 181.644647, "end_time": "2020-08-07T07:23:19.949977", "exception": false, "start_time": "2020-08-07T07:20:18.30533", "status": "completed"} tags=[]
tune_models = []
for m in choice_ms:
    
    m = create_model(m, fold=fold)
    
    tuned_m = tune_model(
        m,
        fold=fold,
        optimize=optimize,
        n_iter=n_iter,
    )
    tune_models.append(tuned_m)

tuned_lightgbm = tune_models[1]
print(tune_models)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#help(finalize_model)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# cv分けず+hold-outのデータも全部含めて学習

#clf1 = imbalance_setup(fold=None)

f_tune_models = []
for m, name in zip(tune_models, choice_ms):
    f_m = finalize_model(m)
    f_tune_models.append(f_m)
    save_model(f_m, model_name=os.path.join(output_dir, "pycaret_tuned_" + name))

print(f_tune_models)

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # 6. Ensemble Model は省略

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # 7. Blend Models

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#help(blend_models)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# CatBoost Classifierはエラーになる

#clf1 = imbalance_setup(fold=fold)
#
#blender = blend_models(estimator_list=tune_models, 
#                       fold=fold,
#                       optimize=optimize,
#                       method="soft", 
#                       choose_better=True,  # 精度改善しなかった場合、create_model()で作ったモデルを返す
#                      )
#save_model(blender, model_name=os.path.join(output_dir, "pycaret_blender"))

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # 8. Stack Models

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#help(stack_models)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#clf1 = imbalance_setup(fold=fold)  # finalize=Trueでもfold指定必要

stacker = stack_models(estimator_list=tune_models[:-1], 
                       meta_model=tune_models[-1],
                       fold=fold,
                       optimize=optimize,
                       #finalize=True,  # cv分けず+hold-outのデータも全部含めて学習
                      )
save_model(stacker, model_name=os.path.join(output_dir, "pycaret_stacker"))

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # 9. Analyze Model

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#from pycaret.classification import *
#help(plot_model)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
plot_model(tune_models[1])

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
plot_model(tune_models[1], plot="confusion_matrix")

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
plot_model(tune_models[1], plot="boundary")

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
plot_model(tune_models[1], plot="feature")  # catboostはエラーになる

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# prだけ異様に時間かかるのでコメントアウト
#plot_model(tune_models[1], plot="pr")

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
plot_model(tune_models[1], plot="class_report")

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
evaluate_model(tune_models[1])

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # 10. Interpret Model

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#catboost = create_model("catboost", cross_validation=False)

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# ## DockerではShapはつかえない。Docker imageが壊れるらしいのでインストールしていない

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#interpret_model(catboost)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#interpret_model(catboost, plot="correlation")

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#interpret_model(catboost, plot="reason", observation=12)

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # 11. AutoML()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# help(automl)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# #%%time
## なんかエラーになる。。。
#automl = automl(optimize=optimize)
#save_model(automl, model_name=os.path.join(output_dir, "pycaret_automl"))
#print(automl)

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # 12. Predict Model

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#pred_holdouts = predict_model(f_tune_models[0])
#pred_holdouts.head()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#new_data = df_test.copy()
#predict_new = predict_model(f_tune_models[0], data=new_data)
#predict_new.head()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]



# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
f_tune_models = []
for name in choice_ms:
    loaded_model = load_model(os.path.join(output_dir, f"pycaret_tuned_{name}"))
    f_tune_models.append(loaded_model)
stacker = load_model(os.path.join(output_dir, f"pycaret_stacker"))


# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
def make_submit(df_test, model, data_dir: str, output_dir: str, csv_name: str):
    """submit csv作成"""
    df_sub = predict_model(model, data=df_test)[["Label"]].reset_index()
    df_sub.columns = ["id", "y"]
    df_sub.to_csv(f"{output_dir}/{csv_name}.csv", index=False)
    display(df_sub.head())
    
    
for m, name in zip(f_tune_models, choice_ms):
    make_submit(df_test, m, data_dir, output_dir, f"{name}_submission")
#make_submit(df_test, blender, data_dir, output_dir, "blender_submission")
make_submit(df_test, stacker, data_dir, output_dir, "stacker_submission")
#make_submit(df_test, automl, data_dir, output_dir, "automl_submission")

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]



# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # 13. Save / Load Model

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#save_model(best, model_name=os.path.join(output_dir, "pycaret_automl"))

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#loaded_bestmodel = load_model(os.path.join(output_dir, "pycaret_automl"))
#print(loaded_bestmodel)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
from sklearn import set_config

set_config(display="diagram")
loaded_bestmodel[0]

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
from sklearn import set_config

set_config(display="text")

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # 14. Deploy Model

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
#deploy_model(best, model_name="best-aws", authentication={"bucket": "pycaret-test"})

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # 15. Get Config / Set Config

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
X_train = get_config("X_train")
X_train.head()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
get_config("seed")

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
from pycaret.classification import set_config

set_config("seed", 999)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
get_config("seed")

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # 16. MLFlow UI

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # !mlflow ui

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "completed"} tags=[]
# # End
# Thank you. For more information / tutorials on PyCaret, please visit https://www.pycaret.org
