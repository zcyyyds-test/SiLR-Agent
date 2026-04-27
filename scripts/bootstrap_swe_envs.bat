@echo off
REM Bootstrap conda envs for SWE-bench Lite repos on Intel server.
REM Runs on Windows + Git Bash shell via WMI. Uses Tsinghua pip+conda mirror
REM because Intel server has no outbound DNS to repo.anaconda.com or pypi.org.
REM Assumes: miniconda3 at C:\Users\Administrator\miniconda3
REM          SWE-bench repo clones already scp'd to D:\zcy\silr-swe-cache\repos\

setlocal
set CONDA=C:\Users\Administrator\miniconda3\Scripts\conda.exe
set CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
set PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
set CACHE=D:\zcy\silr-swe-cache
REM Bump log filename on each bootstrap run (bootstrap.log vs bootstrap2.log)
REM so a zombie conda holding the old file doesn't block the new launch.
set LOG=%CACHE%\bootstrap2.log

if not exist %CACHE%\envs mkdir %CACHE%\envs

echo bootstrap start %DATE% %TIME% > %LOG%

for %%R in (django__django sympy__sympy astropy__astropy scikit-learn__scikit-learn matplotlib__matplotlib sphinx-doc__sphinx pytest-dev__pytest psf__requests pydata__xarray pylint-dev__pylint pallets__flask mwaskom__seaborn) do (
    echo [%%R] creating env >> %LOG%
    REM --override-channels + explicit Tsinghua mirror bypasses conda defaults
    REM (which point at repo.anaconda.com and are blocked by Intel's net).
    call "%CONDA%" create -y -p %CACHE%\envs\%%R --override-channels -c %CONDA_CHANNEL% python=3.11 >> %LOG% 2>&1
    call %CACHE%\envs\%%R\python.exe -m pip install --upgrade pip setuptools wheel >> %LOG% 2>&1
    call %CACHE%\envs\%%R\python.exe -m pip install %CACHE%\repos\%%R >> %LOG% 2>&1
    call %CACHE%\envs\%%R\python.exe -m pip install pytest >> %LOG% 2>&1
    echo [%%R] done >> %LOG%
)
echo all envs ready %DATE% %TIME% >> %LOG%
