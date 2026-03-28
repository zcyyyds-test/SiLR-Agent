"""Platform fixes for Windows Server with many-core CPUs.

Import this module BEFORE importing andes to avoid:
1. multiprocess Pool crash on 128-core Windows (>63 handle limit)
2. OpenBLAS deadlock in scipy on high-core-count machines

Patches applied:
1. OPENBLAS_NUM_THREADS / OMP_NUM_THREADS → 4 (prevent scipy/BLAS deadlock)
2. NUMBA_NUM_THREADS → 4
3. os.cpu_count() → MAX_POOL_WORKERS (ANDES reads this for codegen ncpu)
4. multiprocessing.pool.Pool.__init__ — cap workers
5. multiprocess.pool.Pool.__init__ — cap workers (pathos/sympy)
"""

import os

# --- 1. Environment variables ---
# Critical: OpenBLAS on 128-core Windows deadlocks during scipy.sparse import
# unless thread count is explicitly limited. Must be set BEFORE scipy import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

_MAX_POOL = int(os.environ.get("MAX_POOL_WORKERS", "8"))

# --- 2. Override os.cpu_count() ---
# ANDES System.prepare() reads os.cpu_count() to size its Pool.
# On 128-core Xeon/EPYC this returns 256, exceeding Windows 63-handle limit.
_real_cpu_count = os.cpu_count
os.cpu_count = lambda: _MAX_POOL

# --- 3. Patch multiprocessing (stdlib) ---
try:
    import multiprocessing as _mp_mod
    _mp_mod.cpu_count = lambda: _MAX_POOL

    import multiprocessing.pool as _mp_pool

    _orig_mp_init = _mp_pool.Pool.__init__

    def _patched_mp_init(self, processes=None, *a, **kw):
        if processes is None or processes > _MAX_POOL:
            processes = _MAX_POOL
        _orig_mp_init(self, processes, *a, **kw)

    _mp_pool.Pool.__init__ = _patched_mp_init
except ImportError:
    pass

# --- 4. Patch multiprocess (pathos — used by andes/sympy) ---
try:
    import multiprocess.pool as _mpp_pool

    _orig_mpp_init = _mpp_pool.Pool.__init__

    def _patched_mpp_init(self, processes=None, *a, **kw):
        if processes is None or processes > _MAX_POOL:
            processes = _MAX_POOL
        _orig_mpp_init(self, processes, *a, **kw)

    _mpp_pool.Pool.__init__ = _patched_mpp_init
except ImportError:
    pass
