"""Parallelism helpers for CodonPipe permutation / bootstrap loops.

The hot loops in CodonPipe (Mahal cluster bootstrap, PERMANOVA, gene-set
Aitchison perm, HGT-flag perm, operon co-adaptation null) all share the
same shape: N independent iterations, each consuming a ``np.random``
generator, each producing a small numeric result. This module wraps the
joblib boilerplate plus the two reproducibility safeguards that the
audit report flagged:

  1. **Independent RNG streams per worker.** We seed each iteration's
     generator from a child of ``np.random.SeedSequence(master_seed)``,
     spawned deterministically. The serial run and the parallel run
     therefore produce identical numbers (no global ``np.random.seed()``
     anywhere in the worker code).

  2. **BLAS thread cap inside workers.** When N joblib workers each
     spin up M OpenBLAS threads, you get N*M threads contending for
     the cores. ``threadpoolctl.threadpool_limits(1)`` inside the
     worker context manager caps this at 1 BLAS thread per worker so
     the joblib parallelism is the only parallelism in play.

The number of workers comes from the ``CODONPIPE_JOBS`` environment
variable (defaults to ``-1`` = "all cores"), or can be passed
explicitly. Setting ``CODONPIPE_JOBS=1`` reverts to the serial path —
useful for debugging or when running in a slot-constrained scheduler.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger("codonpipe")


def get_default_n_jobs(default: int = -1) -> int:
    """Resolve the worker count from CODONPIPE_JOBS or fall back to ``default``.

    Returns ``1`` for serial; ``-1`` for "all cores"; or a positive int.
    """
    raw = os.environ.get("CODONPIPE_JOBS")
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "CODONPIPE_JOBS=%r is not an int; falling back to %d.",
            raw, default,
        )
        return default


def spawn_rngs(master_seed: int, n: int) -> list[np.random.Generator]:
    """Return ``n`` independent ``np.random.Generator`` instances seeded
    deterministically from ``master_seed`` via ``SeedSequence.spawn``.

    Using ``SeedSequence`` is the numpy-recommended way to get truly
    independent streams across workers — much better than ``seed=i``
    for ``i in range(n)`` (which shares state across the PCG64 lattice
    and can exhibit subtle correlation).
    """
    ss = np.random.SeedSequence(master_seed)
    return [np.random.default_rng(child) for child in ss.spawn(n)]


def parallel_perm(
    n_iter: int,
    fn: Callable[[np.random.Generator, int], Any],
    master_seed: int,
    n_jobs: int | None = None,
    desc: str = "perm",
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    """Run ``fn`` for ``n_iter`` independent iterations in parallel.

    ``fn`` is called as ``fn(rng, iteration_index, *args, **kwargs)``
    where ``rng`` is a deterministic per-iteration ``np.random.Generator``
    seeded via ``SeedSequence(master_seed).spawn(n_iter)``. The
    iteration index is provided so callers that want stable index-based
    behaviour (e.g. cluster_stability's seed=b in older code) can
    reproduce it; new code should rely on ``rng`` only.

    Returns a list of the per-iteration return values, in iteration
    order.

    Parallelism. Workers are joblib loky processes (process-isolated, so
    the GIL is not a concern), with BLAS threads capped at 1 per worker
    via ``threadpoolctl.threadpool_limits``. Set
    ``CODONPIPE_JOBS=1`` to fall back to a serial loop for debugging
    or when worker startup cost dominates (e.g. n_iter < ~50 and each
    iteration is tiny).
    """
    if n_iter <= 0:
        return []

    if n_jobs is None:
        n_jobs = get_default_n_jobs(default=-1)

    rngs = spawn_rngs(master_seed, n_iter)

    # Serial fast path. Spinning up loky workers for tiny loops costs
    # more than the loop itself; honour CODONPIPE_JOBS=1 and use a plain
    # for-loop. The numerical results are identical because the rngs
    # are spawned the same way.
    if n_jobs == 1 or n_iter < 4:
        return [fn(rngs[i], i, *args, **kwargs) for i in range(n_iter)]

    try:
        from joblib import Parallel, delayed
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        logger.warning(
            "joblib / threadpoolctl unavailable (%s); running %s serially.",
            exc, desc,
        )
        return [fn(rngs[i], i, *args, **kwargs) for i in range(n_iter)]

    def _worker(rng: np.random.Generator, idx: int) -> Any:
        # Cap BLAS threads INSIDE the worker so joblib_workers × BLAS_threads
        # doesn't exceed the physical core count. Each worker still gets
        # joblib's process-level parallelism; what we're disabling is the
        # numpy-internal threading that would multiply on top of it.
        with threadpool_limits(limits=1):
            return fn(rng, idx, *args, **kwargs)

    return Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(_worker)(rngs[i], i) for i in range(n_iter)
    )


def parallel_map(
    items: Sequence[Any],
    fn: Callable[[Any], Any],
    n_jobs: int | None = None,
    desc: str = "map",
    cap_blas: bool = True,
) -> list[Any]:
    """Map ``fn`` over ``items`` in parallel.

    Distinct from :func:`parallel_perm` because there's no per-iteration
    RNG to spawn — use this for things like "render N independent
    figures" or "load N TSV files" where the iteration index alone
    fully determines the work.
    """
    if not items:
        return []
    if n_jobs is None:
        n_jobs = get_default_n_jobs(default=-1)

    if n_jobs == 1 or len(items) < 2:
        return [fn(x) for x in items]

    try:
        from joblib import Parallel, delayed
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        logger.warning(
            "joblib / threadpoolctl unavailable (%s); running %s serially.",
            exc, desc,
        )
        return [fn(x) for x in items]

    def _worker(x: Any) -> Any:
        if cap_blas:
            with threadpool_limits(limits=1):
                return fn(x)
        return fn(x)

    return Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(_worker)(x) for x in items
    )
