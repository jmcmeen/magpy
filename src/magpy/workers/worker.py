"""
Worker -- the one place blocking bioamla calls cross onto a background thread.

Most interactive compute (spectrogram, indices on a view window) is fast enough
to run synchronously. This bridge exists for the operations that are *not*:
``bioamla``'s ``batch_*`` functions, ML inference/training, and catalog
downloads -- all of which already accept an ``on_progress(done, total)``
callback that maps directly onto :attr:`WorkerSignals.progress`.

Cancellation is cooperative. bioamla exposes no ``should_cancel`` parameter, so
the only safe abort point is inside ``on_progress``: when cancellation is
requested we raise :class:`Cancelled` from the progress callback, which unwinds
the bioamla call at the next item boundary. Operations without progress reporting
therefore cannot be cancelled mid-flight -- that is a bioamla limitation, not a
design choice, and callers should not promise otherwise.

Usage::

    worker = Worker(run_batch_detect, files, detector)
    worker.signals.progress.connect(bar.setValue)
    worker.signals.result.connect(on_done)
    worker.signals.error.connect(on_error)
    QThreadPool.globalInstance().start(worker)
"""

from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal


class Cancelled(Exception):
    """Raised inside the progress callback to unwind a cancelled operation."""


class WorkerSignals(QObject):
    """Signals emitted by a :class:`Worker`. Lives on a QObject so they marshal
    back to the thread that owns the connected slots (the UI thread)."""

    started = pyqtSignal()
    progress = pyqtSignal(int, int)  # (done, total)
    result = pyqtSignal(object)  # whatever the wrapped callable returns
    error = pyqtSignal(object)  # the raised Exception
    finished = pyqtSignal()  # always emitted last, success or failure


class Worker(QRunnable):
    """
    Runs ``fn(*args, **kwargs)`` on the global ``QThreadPool`` and reports back
    via :attr:`signals`.

    If ``fn`` accepts an ``on_progress`` keyword, pass ``with_progress=True`` and
    the worker injects a callback that (a) emits :attr:`WorkerSignals.progress`
    and (b) raises :class:`Cancelled` once :meth:`cancel` has been called.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        with_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._with_progress = with_progress
        self._cancel_requested = False

    def cancel(self) -> None:
        """Request cooperative cancellation. Takes effect at the next progress tick."""
        self._cancel_requested = True

    def _on_progress(self, done: int, total: int) -> None:
        if self._cancel_requested:
            raise Cancelled()
        self.signals.progress.emit(done, total)

    def run(self) -> None:  # QRunnable entry point
        self.signals.started.emit()
        try:
            if self._with_progress:
                self._kwargs["on_progress"] = self._on_progress
            result = self._fn(*self._args, **self._kwargs)
        except Cancelled:
            pass  # cancellation is a normal outcome, not an error
        except Exception as exc:  # surface to the UI rather than crash the pool
            self.signals.error.emit(exc)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
