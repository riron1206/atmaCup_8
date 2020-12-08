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
import pandas as pd
pd.set_option('display.max_columns', None)

DATADIR = r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\data\atmacup08-dataset"
train = pd.read_csv(f"{DATADIR}/train.csv")
test = pd.read_csv(f"{DATADIR}/test.csv")
df = pd.concat([train, test], axis=0)
# -

train

test

# +
# 重複データ確認
_df = train.copy()  # 重複削除する場合はcopy不要
print(f"shape before drop: {_df.shape}")
_df.drop_duplicates(inplace=True)
_df.reset_index(drop=True, inplace=True)
print(f"shape after drop: {_df.shape}")

# 欠損データ確認
display(pd.DataFrame({"is_null": _df.isnull().sum()}))

# +
# 重複データ確認
_df = test.copy()  # 重複削除する場合はcopy不要
print(f"shape before drop: {_df.shape}")
_df.drop_duplicates(inplace=True)
_df.reset_index(drop=True, inplace=True)
print(f"shape after drop: {_df.shape}")

# 欠損データ確認
display(pd.DataFrame({"is_null": _df.isnull().sum()}))
# -

# test setには重複レコードあり
test[test.duplicated()]

# 欠損値カラム
_df_miss_sum = pd.DataFrame({"is_null": df.isnull().sum()})
missing_columns = _df_miss_sum[_df_miss_sum["is_null"] > 0].index.to_list()
missing_columns

# # 前処理

# +
import numpy as np

# tbd(確認中)を欠損にする
df["User_Score"] = df["User_Score"].replace('tbd', np.nan)

df["User_Score"].value_counts()

# +
# -1で補完

cate_cols = ["Name", "Platform", "Year_of_Release", "Genre", "Publisher", "Developer", "Rating", ]

for col in cate_cols:
    df[col].fillna(-1, inplace=True)
df


# + code_folding=[]
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
    df, cols_with_missing=["Critic_Score", "Critic_Count", "User_Count",]
)  # 平均値で補間（数値列のみ）

df

# +
# User_Scoreを数値列にする
df["User_Score"] = df["User_Score"].astype('float')

# User_Scoreを文字列にする
df["Year_of_Release"] = df["Year_of_Release"].astype('str')
# -

# 欠損値カラム確認
_df_miss_sum = pd.DataFrame({"is_null": df.isnull().sum()})
missing_columns = _df_miss_sum[_df_miss_sum["is_null"] > 0].index.to_list()
missing_columns

# ラベルエンコディング
cate_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
for col in cate_cols:
    df[col], uni = pd.factorize(df[col])
df

# +
train = df.iloc[: train.shape[0]]
test = df.iloc[train.shape[0] :].reset_index(drop=True)

# 目的変数
sales_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales",]

test = test.drop(sales_cols, axis=1)

train.to_csv("train_preprocess.csv", index=False)
test.to_csv("test_preprocess.csv", index=False)
# -

train

test

# # FE

# +
# 目的変数
sales_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
df_sales = df[sales_cols]
display(df_sales)

df = df.drop(sales_cols, axis=1)
df


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

    df_all[f"{col_name}_sum"] = df.sum(axis=1)
    df_all[f"{col_name}_mean"] = df.mean(axis=1)
    df_all[f"{col_name}_std"] = df.std(axis=1)
    df_all[f"{col_name}_skew"] = df.skew(axis=1)  # 歪度
    
    return df_all

    
df = add_num_row_agg(df, ["Critic_Score", "User_Score"])
df = add_num_row_agg(df, ["Critic_Count", "User_Count"])

display(df)

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

class AggUtil():
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
        #"max",
        #"min",
        "mean",
        "std",  # 標準偏差
        "skew",  # 歪度
        pd.DataFrame.kurt,  # 尖度
    ],
    "Critic_Count": [
        #"max",
        #"min",
        "mean",
        "std",  # 標準偏差
        "skew",  # 歪度
        pd.DataFrame.kurt,  # 尖度
    ],
    "User_Score": [
        #"max",
        #"min",
        "mean",
        "std",  # 標準偏差
        "skew",  # 歪度
        pd.DataFrame.kurt,  # 尖度
    ],
    "Critic_Score": [
        #"max",
        #"min",
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
pd.set_option('display.max_columns', None)
display(df)
# -

df["Critic_Score_*_Critic_Count"] = df["Critic_Score"] * df["Critic_Count"]
df["User_Score_*_User_Count"] = df["User_Score"] * df["User_Count"]
df["Critic_Score_*_User_Score"] = df["Critic_Score"] * df["User_Score"]
df["Critic_Count_+_User_Count"] = df["Critic_Count"] + df["User_Count"]
df["Critic_Count_-_User_Count"] = df["Critic_Count"] - df["User_Count"]
df["Critic_Count_/_all_Count"] = df["Critic_Count"] / df["Critic_Count_+_User_Count"]

# +
from sklearn.cluster import KMeans

def fe_cluster(df, kind, features, n_clusters=100, SEED=42):
    df_ = df[features].copy()
    kmeans_cells = KMeans(n_clusters=n_clusters, random_state=SEED).fit(df_)
    df[f'clusters_{kind}'] = kmeans_cells.predict(df_.values)
    #df = pd.get_dummies(df, columns=[f'clusters_{kind}'])

    return df

df = fe_cluster(df, kind="cate_cols", features=["Name", "Genre", "Publisher", "Developer", "Rating"])
df = fe_cluster(df, kind="Score_Count", features=["Critic_Score", "User_Score", "Critic_Count", "User_Count"])
# -



# +
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import *

class Encoder():
    def __init__(self, 
                 encoder_flags={"count": True, 
                                "target": True, 
                                "catboost": True, 
                                "label": True, 
                                "impute_null": True}) -> None:
        self.encoder_flags = encoder_flags
    
    @staticmethod
    def count_encoder(train_df, valid_df, cat_features=None):
        """
        Count_Encoding: カテゴリ列をカウント値に変換する特徴量エンジニアリング（要はgroupby().size()の集計列追加のこと）
        ※カウント数が同じカテゴリは同じようなデータ傾向になる可能性がある
        https://www.kaggle.com/matleonard/categorical-encodings
        """
        # conda install -c conda-forge category_encoders
        import category_encoders as ce

        if cat_features is None:
            cat_features = train_df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.to_list()

        count_enc = ce.CountEncoder(cols=cat_features)

        # trainだけでfitすること(validationやtest含めるとリークする)
        count_enc.fit(train_df[cat_features])
        train_encoded = train_df.join(
            count_enc.transform(train_df[cat_features]).add_suffix("_count")
        )
        valid_encoded = valid_df.join(
            count_enc.transform(valid_df[cat_features]).add_suffix("_count")
        )

        return train_encoded, valid_encoded

    @staticmethod
    def target_encoder(train_df, valid_df, target_col: str, cat_features=None):
        """
        Target_Encoding: カテゴリ列を目的変数の平均値に変換する特徴量エンジニアリング
        https://www.kaggle.com/matleonard/categorical-encodings
        """
        # conda install -c conda-forge category_encoders
        import category_encoders as ce

        if cat_features is None:
            cat_features = train_df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.to_list()

        target_enc = ce.TargetEncoder(cols=cat_features)

        # trainだけでfitすること(validationやtest含めるとリークする)
        target_enc.fit(train_df[cat_features], train_df[target_col])

        train_encoded = train_df.join(
            target_enc.transform(train_df[cat_features]).add_suffix("_target")
        )
        valid_encoded = valid_df.join(
            target_enc.transform(valid_df[cat_features]).add_suffix("_target")
        )
        return train_encoded, valid_encoded

    @staticmethod
    def catboost_encoder(train_df, valid_df, target_col: str, cat_features=None):
        """
        CatBoost_Encoding: カテゴリ列を目的変数の1行前の行からのみに変換する特徴量エンジニアリング
        CatBoost使ったターゲットエンコーディング
        https://www.kaggle.com/matleonard/categorical-encodings
        """
        # conda install -c conda-forge category_encoders
        import category_encoders as ce

        if cat_features is None:
            cat_features = train_df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.to_list()

        cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=7)

        # trainだけでfitすること(validationやtest含めるとリークする)
        cb_enc.fit(train_df[cat_features], train_df[target_col])

        train_encoded = train_df.join(
            cb_enc.transform(train_df[cat_features]).add_suffix("_cb")
        )
        valid_encoded = valid_df.join(
            cb_enc.transform(valid_df[cat_features]).add_suffix("_cb")
        )
        return train_encoded, valid_encoded

    @staticmethod
    def impute_null_add_flag_col(df, strategy="median", cols_with_missing=None, fill_value=None):
        """欠損値を補間して欠損フラグ列を追加する
        fill_value はstrategy="constant"の時のみ有効になる補間する定数
        """
        from sklearn.impute import SimpleImputer

        df_plus = df.copy()

        if cols_with_missing is None:
            if strategy in ["median", "median"]:
                # 数値列で欠損ある列探す
                cols_with_missing = [col for col in df.columns if (df[col].isnull().any()) and (df[col].dtype.name not in ["object", "category", "bool"])]
            else:
                # 欠損ある列探す
                cols_with_missing = [col for col in df.columns if (df[col].isnull().any())]

        for col in cols_with_missing:
            # 欠損フラグ列を追加
            df_plus[col + "_was_missing"] = df[col].isnull()
            df_plus[col + "_was_missing"] = df_plus[col + "_was_missing"].astype(int)
            # 欠損値を平均値で補間
            my_imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            df_plus[col] = my_imputer.fit_transform(df[cols_with_missing])

        return df_plus
    
    def run_encoders(self, X, y, train_index, val_index):
        """カウント,ターゲット,CatBoost,ラベルエンコディング一気にやる。cvの処理のforの中に書きやすいように"""
        train_df = pd.concat([X, y], axis=1)
        t_fold_df, v_fold_df = train_df.iloc[train_index], train_df.iloc[val_index]

        if self.encoder_flags["count"]:
            # カウントエンコディング
            t_fold_df, v_fold_df = Encoder().count_encoder(t_fold_df, v_fold_df, cat_features=None)
        if self.encoder_flags["target"]:
            # ターゲットエンコディング
            t_fold_df, v_fold_df = Encoder().target_encoder(t_fold_df, v_fold_df, target_col=target_col, cat_features=None)
        if self.encoder_flags["catboost"]:
            # CatBoostエンコディング
            t_fold_df, v_fold_df = Encoder().catboost_encoder(t_fold_df, v_fold_df, target_col=target_col, cat_features=None)
        
        if self.encoder_flags["label"]:
            # ラベルエンコディング
            train_df = t_fold_df.append(v_fold_df)  # trainとval再連結
            cate_cols = t_fold_df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
            for col in cate_cols:
                train_df[col], uni = pd.factorize(train_df[col])
                
        if self.encoder_flags["impute_null"]:
            # 欠損置換（ラベルエンコディングの後じゃないと処理遅くなる）
            nulls = train_df.drop(y.name, axis=1).isnull().sum().to_frame()
            null_indexs = [index for index, row in nulls.iterrows() if row[0] > 0]
            train_df = Encoder().impute_null_add_flag_col(train_df, cols_with_missing=null_indexs, strategy="most_frequent")  # 最頻値で補間
        
        t_fold_df, v_fold_df = train_df.iloc[train_index], train_df.iloc[val_index]
        print(
            "run encoding Train shape: {}, valid shape: {}".format(
                t_fold_df.shape, v_fold_df.shape
            )
        )
        feats = t_fold_df.columns.to_list()
        feats.remove(target_col)
        X_train, y_train = (t_fold_df[feats], t_fold_df[target_col])
        X_val, y_val = (v_fold_df[feats], v_fold_df[target_col])
        return X_train, y_train, X_val, y_val
    

if __name__ == '__main__':
    # サンプルデータ
    import seaborn as sns
    df = sns.load_dataset("titanic")
    df = df.drop(["alive"], axis=1)
    target_col = "survived"
    feats = df.columns.to_list()
    feats.remove(target_col)
    print(df.shape)
    
    # KFoldで実行する場合
    folds = KFold(n_splits=4, shuffle=True, random_state=1001)
    for n_fold, (train_idx, valid_idx) in tqdm(enumerate(folds.split(df[feats], df[target_col]))):
        # カウント,ターゲット,CatBoost,ラベルエンコディング一気にやる
        X_train, y_train, X_val, y_val = Encoder().run_encoders(df[feats], df[target_col], train_idx, valid_idx)
        train_df = pd.concat([X_train, y_train], axis=1)
        valid_df = pd.concat([X_val, y_val], axis=1)
        print(f"fold: {n_fold}:", train_df.shape, valid_df.shape)
    
    # test setでする場合
    X_train, y_train, X_test, y_test = Encoder().run_encoders(df[feats], df[target_col], list(range(700)), list(range(700, df.shape[0])))
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    print(train_df.shape, test_df.shape)
# -





# +
df.info()
display(df.head().style.background_gradient(cmap="Pastel1"))
display(df.describe().T.style.background_gradient(cmap="Pastel1"))

# カラム名 / カラムごとのユニーク値数 / 最も出現頻度の高い値 / 最も出現頻度の高い値の出現回数 / 欠損損値の割合 / 最も多いカテゴリの割合 / dtypes を表示
stats = []
for col in df.columns:
    stats.append(
        (
            col,
            df[col].nunique(),
            df[col].value_counts().index[0],
            df[col].value_counts().values[0],
            df[col].isnull().sum() * 100 / df.shape[0],
            df[col].value_counts(normalize=True, dropna=False).values[0] * 100,
            df[col].dtype,
        )
    )
stats_df = pd.DataFrame(
    stats,
    columns=[
        "Feature",
        "Unique values",
        "Most frequent item",
        "Freuquence of most frequent item",
        "Percentage of missing values",
        "Percentage of values in the biggest category",
        "Type",
    ],
)
display(stats_df.sort_values("Percentage of missing values", ascending=False).style.background_gradient(cmap="Pastel1"))
# -



train = df.iloc[: train.shape[0]]
test = df.iloc[train.shape[0] :].reset_index(drop=True)

train

test
