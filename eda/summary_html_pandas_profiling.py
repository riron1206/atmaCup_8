# coding: utf-8
"""
pandas_profilingでテーブル形式ファイルの統計情報をhtmlで出力する
htmlは以下の項目を表示
- OverView:データサマリー
- Variables:列各の統計情報。ヒストグラムも表示する
- Correlation:各列のピアソン/スピアマンの相関係数
- Sample:実データ

Usage:
    # tsvのサマリーhtmlをtmpディレクトリに出力
    $ python summary_html_pandas_profiling.py -o tmp -i train.csv

    # エクセルの各シートのサマリーhtmlをtmpディレクトリに出力。シートごとにhtml出力。n_skiprowで各シートの先頭8行飛ばす。is_minimalで散布図など書かないようにする
    $ python summary_html_pandas_profiling.py \
        --output_dir tmp \
        --input_file tmp.xlsx' \
        --n_skiprow 8 \
        --is_minimal
"""
import os
import argparse
import pandas as pd
import pandas_profiling as pdp
import pathlib
import openpyxl
import warnings

warnings.filterwarnings("ignore")

# print("pdp.__version__", pdp.__version__)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input_file", type=str, required=True, help="input file path."
    )
    ap.add_argument(
        "-o", "--output_dir", type=str, default=None, help="html output dir path."
    )
    ap.add_argument(
        "-n_s", "--n_skiprow", type=int, default=None, help="number of rows to skip."
    )
    ap.add_argument(
        "-n_h", "--n_header", type=int, default=0, help="pd.read_csv arg header."
    )
    ap.add_argument(
        "-is_m",
        "--is_minimal",
        action="store_const",
        const=True,
        default=False,
        help="ProfileReport minimal flag.",
    )
    args = ap.parse_args()

    path = args.input_file

    if args.output_dir is None:
        output_dir = "."
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = args.output_dir

    if args.n_skiprow is None:
        skiprows = None
    else:
        skiprows = range(args.n_skiprow)

    output_file = os.path.join(
        output_dir, str(pathlib.Path(path).name) + "_profile.html"
    )

    if str(pathlib.Path(path).suffix) in [".txt", ".tsv"]:
        df = pd.read_csv(path, sep="\t", skiprows=skiprows, header=args.n_header)

        if df.shape[0] > 200000 or df.shape[1] > 100:
            args.is_minimal = True

        try:
            pdp.ProfileReport(df, minimal=args.is_minimal).to_file(
                output_file=output_file
            )
        except Exception as e:
            print("ERROR:", e)
            print("INFO:", "Re exe minimal=True")
            pdp.ProfileReport(df, minimal=True).to_file(output_file=output_file)

    if str(pathlib.Path(path).suffix) in [".csv"]:
        df = pd.read_csv(path, skiprows=skiprows, header=args.n_header)

        if df.shape[0] > 200000 or df.shape[1] > 100:
            args.is_minimal = True

        try:
            pdp.ProfileReport(df, minimal=args.is_minimal).to_file(
                output_file=output_file
            )
        except Exception as e:
            print("ERROR:", e)
            print("INFO:", "Re exe minimal=True")
            pdp.ProfileReport(df, minimal=True).to_file(output_file=output_file)

    if str(pathlib.Path(path).suffix) in [".xlsx", ".xlsm", ".xlsb", ".xls"]:
        book = openpyxl.load_workbook(path)
        sheets = book.sheetnames  # シート名をすべて取得
        # print(sheets)
        for s in sheets:
            output_file = os.path.join(
                output_dir, str(pathlib.Path(path).name) + "_" + s + "_profile.html"
            )
            try:
                df = pd.read_excel(
                    path, sheet_name=s, skiprows=skiprows, header=args.n_header
                )

                if df.shape[0] > 200000 or df.shape[1] > 100:
                    args.is_minimal = True

                # display(df.head())
                for col in df.columns.tolist():
                    if "Unnamed" in col:
                        df = df.drop(col, axis=1)
                pdp.ProfileReport(df, minimal=args.is_minimal).to_file(
                    output_file=output_file
                )
            except Exception as e:
                print("ERROR:", e)
                print("INFO:", "Re exe minimal=True")
                pdp.ProfileReport(df, minimal=True).to_file(output_file=output_file)
