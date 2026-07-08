from __future__ import annotations

from contextlib import contextmanager
from queue import Empty
from threading import Thread
from typing import Iterator

from joblib import parallel
from tqdm.auto import tqdm


def progress_bar(*args, **kwargs) -> tqdm:
    """Create a consistently configured tqdm progress bar."""
    kwargs.setdefault("ascii", True)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", True)
    return tqdm(*args, **kwargs)


@contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> Iterator[tqdm]:
    """Update a tqdm bar when joblib parallel batches complete."""
    old_callback = parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(old_callback):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


class ProgressMonitor:
    """Owns terminal progress bars in the main process."""

    def __init__(
        self,
        queue,
        *,
        total_stages: int,
        total_samples: int = 0,
        enabled: bool = True,
    ):
        self.queue = queue
        self.enabled = enabled
        self._thread: Thread | None = None
        self.stage_bar = None
        self.sample_bar = None
        self.total_stages = total_stages
        self.total_samples = total_samples

    def __enter__(self):
        if not self.enabled or self.queue is None:
            return self

        self.stage_bar = progress_bar(
            total=self.total_stages,
            desc="pipeline",
            unit="step",
            position=0,
        )
        self.sample_bar = progress_bar(
            total=self.total_samples,
            desc="explanations",
            unit="sample",
            position=1,
        )
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled or self.queue is None:
            return

        self.queue.put({"type": "stop"})
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.sample_bar is not None:
            self.sample_bar.close()
        if self.stage_bar is not None:
            self.stage_bar.close()

    def _run(self) -> None:
        while True:
            try:
                event = self.queue.get(timeout=0.2)
            except Empty:
                continue

            event_type = event.get("type")
            if event_type == "stop":
                break

            if event_type == "active" and self.stage_bar is not None:
                label = event.get("label")
                if label:
                    self.stage_bar.set_postfix_str(f"running: {label}", refresh=True)

            elif event_type == "stage" and self.stage_bar is not None:
                self.stage_bar.update(event.get("amount", 1))
                label = event.get("label")
                if label:
                    self.stage_bar.set_postfix_str(f"done: {label}", refresh=True)

            elif event_type == "sample_total":
                continue

            elif event_type == "sample" and self.sample_bar is not None:
                self.sample_bar.update(event.get("amount", 1))
                label = event.get("label")
                if label:
                    self.sample_bar.set_postfix_str(f"last: {label}", refresh=True)
