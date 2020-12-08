@rem çÏê¨ì˙2020/07/08 summary_html_pandas_profiling.pyé¿çs

rem call activate tfgpu

set PY_DIR=C:\Users\81908\MyGitHub\summary_html_pandas_profiling\

@rem call python %PY_DIR%summary_html_pandas_profiling.py ^
call poetry run python %PY_DIR%summary_html_pandas_profiling.py ^
-o output ^
-i C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\data\atmacup08-dataset\train.csv

@rem call python %PY_DIR%summary_html_pandas_profiling.py ^
call poetry run python %PY_DIR%summary_html_pandas_profiling.py ^
-o output ^
-i C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\atmaCup_#8\data\atmacup08-dataset\test.csv

pause