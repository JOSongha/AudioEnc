"""DataLoader wrapper that survives a worker SIGKILL.

The native PyTorch DataLoader raises ``RuntimeError: DataLoader worker (pid X)
is killed by signal: ...`` when one of its worker subprocesses dies (OOM kill,
SIGKILL from a watchdog, hung-and-then-reaped). The error propagates to the
main process and there is no built-in respawn — the training loop crashes.

For our use case (whisper-tiny v4 with audio resample + log-mel in worker),
worker stalls are not just possible but periodic (NCCL collective timeout
observed at step 12k and again at 24k). Killing-then-respawning the offending
worker requires either custom multiprocessing scaffolding (large) or
recreating the entire DataLoader instance (this module's choice).

What this wrapper does:

  * On the first ``__iter__`` it builds the DataLoader via ``build_fn``.
  * Each ``__next__`` is wrapped in ``try / except``. A worker-death exception
    triggers tearing down the DataLoader, garbage-collecting + emptying the
    CUDA cache, and rebuilding from scratch with a fresh seed offset. The
    next ``__next__`` call then returns from the rebuilt iterator.
  * For streaming datasets the rebuild starts the iterator from the beginning
    of the stream — same data revisited, but training-progress (model
    weights, optimizer state, step counter) is preserved by the surrounding
    Trainer state.

Caveats:

  * **NCCL collective timeout must be wider than the rebuild wall time.**
    While one rank rebuilds (~5-30 s for cold dataloader spin-up), peer ranks
    are still mid-iteration and will eventually hit the gradient allreduce
    and stall waiting for the rebuilding rank. With the default 10-minute
    NCCL watchdog this is fine; if the rebuild can take longer in your
    environment, bump ``ddp_timeout`` and / or call
    ``patch_default_pg_timeout`` from ``launcher.py``.
  * Rebuild is per-rank: only the rank whose worker died rebuilds. Peer ranks
    keep their existing dataloader.
  * Attribute access (``dataloader.dataset`` etc.) is forwarded to the inner
    DataLoader so the Trainer's progress / step-count math keeps working.
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Iterator, Optional


_WORKER_HINTS = (
    "dataloader worker",  # the canonical phrase
    "worker process",
)
_DEATH_HINTS = (
    "killed by signal",
    "signal: killed",
    "received signal",
    "exitcode",
    "exited unexpectedly",  # PyTorch outer message after a worker died
    "broken pipe",
)


def _msg_chain(exc: BaseException) -> str:
    """Concatenate the message of `exc` with all chained `__cause__` /
    `__context__` messages, so that nested RuntimeErrors with the
    informative phrase deeper in the chain still get matched."""
    parts = []
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(str(cur))
        cur = cur.__cause__ or cur.__context__
    return " || ".join(parts).lower()


def _looks_like_worker_death(exc: BaseException) -> bool:
    """Decide whether an exception came from a dead DataLoader worker.

    PyTorch raises a wrapper RuntimeError with the message ``DataLoader
    worker (pid(s) X) exited unexpectedly`` while the underlying signal
    (``killed by signal: ...``) is buried in ``exc.__cause__``. Walk the
    cause chain so either layer can match.

    Other RuntimeError / OSError instances are re-raised so genuine bugs
    don't get silently retried.
    """
    msg = _msg_chain(exc)
    if "stopiteration" in msg:
        return False
    has_worker = any(h in msg for h in _WORKER_HINTS)
    has_death = any(h in msg for h in _DEATH_HINTS)
    return has_worker and has_death


class _FTIterator(Iterator[Any]):
    def __init__(
        self,
        build_fn: Callable[[], Any],
        dl: Any,
        max_retries: int,
        log: Callable[[str], None],
    ) -> None:
        self._build_fn = build_fn
        self._dl = dl
        self._it = iter(dl)
        self._max_retries = max_retries
        self._retries = 0
        self._log = log

    def __iter__(self) -> "_FTIterator":
        return self

    def __next__(self) -> Any:
        while True:
            try:
                return next(self._it)
            except StopIteration:
                raise
            except (RuntimeError, OSError, EOFError, ConnectionResetError) as exc:
                if not _looks_like_worker_death(exc):
                    raise
                self._retries += 1
                self._log(
                    f"[FaultTolerantDataLoader] worker died "
                    f"(attempt {self._retries}/{self._max_retries}): "
                    f"{type(exc).__name__}: {exc}"
                )
                if self._retries > self._max_retries:
                    raise RuntimeError(
                        f"DataLoader rebuild exhausted "
                        f"(max_retries={self._max_retries})"
                    ) from exc
                self._teardown_and_rebuild()

    def _teardown_and_rebuild(self) -> None:
        # Free any references the dead worker pool still owns.
        try:
            del self._it
        except Exception:
            pass
        try:
            del self._dl
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._dl = self._build_fn()
        self._it = iter(self._dl)


class FaultTolerantDataLoader:
    """DataLoader-shaped wrapper that rebuilds the underlying DataLoader
    when a worker subprocess dies.

    Usage:

        def build():
            return torch.utils.data.DataLoader(...)

        dl = FaultTolerantDataLoader(build, max_retries=5)

    Looks like a DataLoader from the outside: ``len``, ``iter``, attribute
    access on ``.dataset`` / ``.batch_size`` / ``.dataloader_persistent_workers``
    all forward to the wrapped instance.
    """

    def __init__(
        self,
        build_fn: Callable[[], Any],
        max_retries: int = 5,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._build_fn = build_fn
        self._max_retries = max_retries
        self._logger = logger or (lambda msg: print(msg, flush=True))
        self._dl = build_fn()

    def __iter__(self) -> _FTIterator:
        return _FTIterator(self._build_fn, self._dl, self._max_retries, self._logger)

    def __len__(self) -> int:
        return len(self._dl)

    # Forward everything else to the wrapped DataLoader so callers that
    # poke at dataloader.dataset / .batch_size / .num_workers / etc. just work.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._dl, name)


def patch_default_pg_timeout(seconds: int = 3600) -> None:
    """Raise the torch.distributed default process-group timeout.

    This is the timer that NCCL's watchdog uses to detect a stalled collective.
    Default is 600 s (10 min); we bump it so that a per-rank DataLoader
    rebuild has time to complete before peer ranks declare the rebuilding
    rank dead.

    The plain Python rebind of ``_DEFAULT_PG_NCCL_TIMEOUT`` is *not* enough — it
    only updates the Python alias; the C++ side keeps the original 600 s
    constant which sub-PG creation reads. The reliable fix is to wrap
    ``dist.init_process_group`` and ``dist.new_group`` so every PG (incl. the
    DeepSpeed ZeRO-2 ``dp_process_group``) gets an explicit ``timeout`` even
    when the caller forgets to pass one.

    Idempotent. Call once near the top of ``launcher.py`` *before* any code
    triggers ``torch.distributed.init_process_group``.
    """
    try:
        import datetime
        import torch.distributed as dist
        import torch.distributed.distributed_c10d as c10d
    except Exception:
        return
    new = datetime.timedelta(seconds=int(seconds))
    # Public name in newer torch versions; private fallback for older.
    for attr in ("default_pg_timeout", "_DEFAULT_PG_TIMEOUT"):
        if hasattr(c10d, attr):
            setattr(c10d, attr, new)
    # Some torch versions also keep a per-backend default.
    try:
        c10d._DEFAULT_PG_NCCL_TIMEOUT = new
    except Exception:
        pass

    # Wrap init_process_group and new_group so callers without an explicit
    # ``timeout`` (DeepSpeed ZeRO-2 dp_process_group, accelerate, etc.) still
    # get the bumped value. Mark with a sentinel so we don't double-wrap on
    # repeated calls.
    def _wrap(func, name):
        if getattr(func, "_lf_pg_timeout_wrapped", False):
            return func

        def wrapped(*args, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = new
            return func(*args, **kwargs)

        wrapped._lf_pg_timeout_wrapped = True
        wrapped.__wrapped__ = func
        wrapped.__name__ = name
        return wrapped

    if hasattr(dist, "init_process_group"):
        dist.init_process_group = _wrap(dist.init_process_group, "init_process_group")
        c10d.init_process_group = dist.init_process_group
    if hasattr(dist, "new_group"):
        dist.new_group = _wrap(dist.new_group, "new_group")
        c10d.new_group = dist.new_group


def install_grad_nan_guard(model) -> int:
    """Sanitize NaN / Inf gradients in-place before any collective sees them.

    Why: a NaN gradient on one rank causes NCCL reduce-scatter on that rank to
    enter a bad state and never complete. Other ranks then stall at the next
    collective and trip the 600 s timeout. Stack trace at hang shows DeepSpeed
    ``mask_nan_or_inf_with_val_inplace`` (Whisper-small v4 Stage1, step 12850,
    deterministic across re-runs).

    The hook fires post-accumulation per parameter, so the gradient is sanitized
    before DeepSpeed's reduce-scatter / allreduce.

    Returns the number of hooks installed.
    """
    try:
        import torch
    except Exception:
        return 0
    n = 0

    def _hook(p):
        if p.grad is None:
            return
        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

    for p in model.parameters():
        if not p.requires_grad:
            continue
        # Newer torch only; if the API isn't there we silently skip.
        reg = getattr(p, "register_post_accumulate_grad_hook", None)
        if reg is None:
            continue
        reg(_hook)
        n += 1
    return n
