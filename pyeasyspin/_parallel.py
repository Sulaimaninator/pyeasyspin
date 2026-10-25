"""Multicore support: BLAS thread limits for single simulations, and a process pool for
independent simulations (esfit's model evaluations, parameter scans).

Settings (pyeasyspin.config):
  Workers          number of worker processes; 'auto' (default) = all CPU cores, 1 = serial
  BLASThreads      BLAS threads inside a pyeasyspin call (default 1: EPR matrices are small, and
                   multithreaded BLAS on them is slower); None leaves the library default
  ParallelMinTime  run a batch in parallel only if it would take longer than this serially (s)

The pool is started on first use and kept alive between calls (it shuts down after 5 idle
minutes), so its start-up cost (a few seconds on Windows) is paid once per session."""
import os
import sys
import threading
import time
import warnings

import numpy as np

from .config import config

_local = threading.local()
_controller = None
_controller_nmods = -1
_executor = None
_executor_workers = 0
_broken = False

_WORKER_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'PYEASYSPIN_WORKER': '1'}


def n_workers():
    """Number of worker processes from config.Workers (1 inside a worker: pools never nest)."""
    if os.environ.get('PYEASYSPIN_WORKER'):
        return 1
    w = config.Workers
    if w is None or w == 'auto':
        return os.cpu_count() or 1
    return max(1, int(w))


# --- BLAS threads ---------------------------------------------------------------------------

def _get_controller():
    global _controller, _controller_nmods
    n = len(sys.modules)
    if _controller is None or n != _controller_nmods:
        from threadpoolctl import ThreadpoolController
        _controller = ThreadpoolController()
        _controller_nmods = n
    return _controller


class blas_threads:
    """Context manager: limits BLAS threads to config.BLASThreads for the outermost pyeasyspin
    call (nested calls see depth > 0 and leave the setting alone)."""

    def __enter__(self):
        self._depth = getattr(_local, 'depth', 0)
        _local.depth = self._depth + 1
        self._limiter = None
        if self._depth == 0 and config.BLASThreads is not None:
            self._limiter = _get_controller().limit(limits=int(config.BLASThreads), user_api='blas')
        return self

    def __exit__(self, *exc):
        if self._limiter is not None:
            self._limiter.restore_original_limits()
        _local.depth = self._depth
        return False


# --- process pool ---------------------------------------------------------------------------

def _get_executor(workers):
    global _executor, _executor_workers
    if _executor is None or _executor_workers != workers:
        from loky import get_reusable_executor
        _executor = get_reusable_executor(max_workers=workers, timeout=300, env=_WORKER_ENV)
        _executor_workers = workers
    return _executor


def pool_running():
    return _executor is not None


class _Raised:
    """An exception raised by a model call, carried back to be re-raised where it is used."""

    def __init__(self, exc):
        self.exc = exc


def _call(fcn, args, kw):
    try:
        return fcn(*args, **kw)
    except Exception as e:          # returned, not raised: the caller decides when it matters
        return _Raised(e)


def _call_chunk(fcn, arglist, kw):
    t = time.process_time()
    with blas_threads():
        res = [_call(fcn, a, kw) for a in arglist]
    return res, time.process_time() - t


_worker_cpu = 0.0


def cpu_time():
    """CPU time of this process plus the CPU time its workers spent on its tasks (s). esfit's
    algorithms measure Opt.maxTime with it, as EasySpin does with MATLAB's cputime."""
    return time.process_time() + _worker_cpu


def worthwhile(n_tasks, t_task):
    """Whether n_tasks evaluations of about t_task seconds each should go to the pool."""
    w = n_workers()
    if _broken or w < 2 or n_tasks < 2 or t_task is None:
        return False
    min_time = float(config.ParallelMinTime)
    if not pool_running():
        min_time = max(min_time, 2.0)       # first use also pays for starting the workers
    return n_tasks * t_task > min_time


def run(fcn, arglist, kw=None):
    """[fcn(*args, **kw) for args in arglist] on the process pool, in order. Exceptions raised by
    fcn come back as _Raised objects. Returns None if the pool cannot be used (then the caller
    computes serially)."""
    global _broken, _worker_cpu
    kw = kw or {}
    w = n_workers()
    n = len(arglist)
    nchunks = min(n, 4 * w)
    bounds = [round(i * n / nchunks) for i in range(nchunks + 1)]
    chunks = [arglist[bounds[i]:bounds[i + 1]] for i in range(nchunks)]
    try:
        ex = _get_executor(w)
        futures = [ex.submit(_call_chunk, fcn, c, kw) for c in chunks if c]
        out = []
        for f in futures:
            res, cpu = f.result()
            out.extend(res)
            _worker_cpu += cpu
        return out
    except Exception as e:          # pickling problems, a crashed worker, ...
        _broken = True
        warnings.warn(f'pyeasyspin: parallel evaluation unavailable ({type(e).__name__}: {e}); '
                      'continuing serially', RuntimeWarning, stacklevel=3)
        return None


def chunked(fcn, n, probe=8):
    """Run fcn(indices) over the indices 0..n-1 in consecutive chunks and return the chunk results
    in order. The first `probe` indices run serially and are timed; the rest go to the pool in
    chunks if that is worthwhile, otherwise they run serially as one chunk. fcn must give results
    that the caller can merge (each chunk sees only its own indices)."""
    k = min(n, probe)
    t = time.perf_counter()
    out = [fcn(np.arange(k))]
    dt = time.perf_counter() - t
    if k < n:
        rest = np.arange(k, n)
        nch = min(rest.size, 4 * n_workers())
        res = None
        if worthwhile(nch, dt / k * rest.size / nch):
            res = run(fcn, [(c,) for c in np.array_split(rest, nch)])
        if res is None:
            out.append(fcn(rest))
        else:
            for r in res:
                if isinstance(r, _Raised):
                    raise r.exc
            out.extend(res)
    return out


def prefetch(fcn, points):
    """Ask a model wrapper that supports it (esfit's residual and rmsd functions) to compute
    the given parameter points in parallel ahead of use. A no-op for other functions."""
    p = getattr(fcn, 'prefetch', None)
    if p is not None and len(points) > 1:
        p(points)


def batch_size(fcn):
    """How many points an algorithm should prepare ahead for prefetch(); 1 = no prefetching."""
    if getattr(fcn, 'prefetch', None) is None or _broken or n_workers() < 2:
        return 1
    return 4 * n_workers()


def parallel_map(fcn, inputs, workers=None):
    """Evaluate fcn for every element of inputs, on all CPU cores.

    Each element of inputs is a tuple of positional arguments (a non-tuple is passed as the
    single argument). Returns the list of results in input order. Example, a scan over g:

        systems = [es.Struct(g=g, lwpp=1.0) for g in np.linspace(1.98, 2.02, 40)]
        spectra = es.parallel_map(es.pepper, [(s, Exp) for s in systems])

    workers overrides config.Workers. Exceptions raised by fcn are re-raised here."""
    arglist = [a if isinstance(a, tuple) else (a,) for a in inputs]
    old = config.Workers
    if workers is not None:
        config.Workers = workers
    try:
        res = run(fcn, arglist) if n_workers() > 1 and len(arglist) > 1 else None
    finally:
        config.Workers = old
    if res is None:
        with blas_threads():
            res = [_call(fcn, a, {}) for a in arglist]
    for r in res:
        if isinstance(r, _Raised):
            raise r.exc
    return res


class ModelCache:
    """Model outputs computed ahead of use, keyed by the parameter vector.

    compute(keys, arglist) evaluates the model for the given arguments in parallel when that is
    worthwhile and stores the results; take(key) returns and removes a stored result (re-raising
    a stored exception), or None. timing() records how long serial evaluations take, which
    decides whether a batch is worth sending to the pool."""

    def __init__(self, fcn, kw):
        self.fcn = fcn
        self.kw = kw
        self.store = {}
        self.t_eval = None

    def timing(self, dt):
        self.t_eval = dt if self.t_eval is None else 0.7 * self.t_eval + 0.3 * dt

    def compute(self, keys, arglist):
        todo = [(k, a) for k, a in zip(keys, arglist) if k not in self.store]
        seen = set()
        todo = [(k, a) for k, a in todo if not (k in seen or seen.add(k))]
        if len(todo) < 2 or n_workers() < 2:
            return
        if self.t_eval is None:                # time the first point here, then decide for the rest
            k, a = todo.pop(0)
            t = time.perf_counter()
            self.store[k] = _call(self.fcn, a, self.kw)
            self.timing(time.perf_counter() - t)
        if len(todo) < 2 or not worthwhile(len(todo), self.t_eval):
            return
        res = run(self.fcn, [a for _, a in todo], self.kw)
        if res is None:
            return
        for (k, _), r in zip(todo, res):
            self.store[k] = r

    def take(self, key):
        r = self.store.pop(key, None)
        if isinstance(r, _Raised):
            raise r.exc
        return r
