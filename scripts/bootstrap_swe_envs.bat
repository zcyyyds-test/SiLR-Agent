@echo off
REM Bootstrap one conda env per SWE-bench Lite repo (12 repos total) so the
REM verifier can run pytest against patched code without leaking imports
REM across instances.
REM
REM Required env vars:
REM   CONDA       absolute path to conda.exe (e.g. ...\miniconda3\Scripts\conda.exe)
REM   SWE_CACHE   working directory holding repos/<owner>__<repo>/ source clones
REM Optional (override the defaults if your network needs it):
REM   CONDA_CHANNEL    defaults to the Tsinghua main mirror
REM   PIP_INDEX_URL    defaults to the Tsinghua pypi mirror
REM   PIP_TRUSTED_HOST defaults to pypi.tuna.tsinghua.edu.cn

setlocal
if "%CONDA_CHANNEL%"=="" set CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
if "%PIP_INDEX_URL%"=="" set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
if "%PIP_TRUSTED_HOST%"=="" set PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

REM Bump log filename on each bootstrap run so a zombie conda holding the old
REM file doesn't block the new launch.
set LOG=%SWE_CACHE%\bootstrap2.log

if not exist %SWE_CACHE%\envs mkdir %SWE_CACHE%\envs

echo bootstrap start %DATE% %TIME% > %LOG%

for %%R in (django__django sympy__sympy astropy__astropy scikit-learn__scikit-learn matplotlib__matplotlib sphinx-doc__sphinx pytest-dev__pytest psf__requests pydata__xarray pylint-dev__pylint pallets__flask mwaskom__seaborn) do (
    echo [%%R] creating env >> %LOG%
    REM --override-channels + explicit mirror bypasses conda defaults
    REM (which point at repo.anaconda.com, often blocked in restricted networks).
    call "%CONDA%" create -y -p %SWE_CACHE%\envs\%%R --override-channels -c %CONDA_CHANNEL% python=3.11 >> %LOG% 2>&1
    call %SWE_CACHE%\envs\%%R\python.exe -m pip install --upgrade pip setuptools wheel >> %LOG% 2>&1
    call %SWE_CACHE%\envs\%%R\python.exe -m pip install %SWE_CACHE%\repos\%%R >> %LOG% 2>&1
    call %SWE_CACHE%\envs\%%R\python.exe -m pip install pytest >> %LOG% 2>&1
    echo [%%R] done >> %LOG%
)
echo all envs ready %DATE% %TIME% >> %LOG%
