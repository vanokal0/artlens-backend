"""Bounded Windows-spawn preprocessing of independent images, without TensorFlow."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path

from artlens.prepare import digest
from artlens.preprocessing import ann_features, cnn_tensor


def preprocess_worker(task):
    root, sample, model, verify_hash = task
    try:
        path = Path(root) / sample.filename
        if verify_hash and digest(path) != sample.sha256:
            raise ValueError("Source image checksum changed")
        function = {"simple_ann": ann_features, "cnn": cnn_tensor}[model]
        return sample, function(path)
    except Exception as exc:
        raise RuntimeError(f"Preprocessing failed for {sample.filename} ({sample.identity}): {exc}") from exc


def iter_preprocessed(samples, root, model="simple_ann", workers=1, max_in_flight=None, verify_hash=True):
    """Yield (Sample, array) in input order. At most 2*workers tasks by default.

    Paths/metadata go to workers, only compact processed arrays come back. Closing
    the generator cancels pending tasks and waits for active workers to finish.
    """
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4 on this machine")
    if model not in {"simple_ann", "cnn"}:
        raise ValueError("Unknown model")
    bound = max_in_flight if max_in_flight is not None else 2 * workers
    if bound < 1 or bound > 8:
        raise ValueError("max_in_flight must be between 1 and 8")
    tasks = ((str(root), sample, model, verify_hash) for sample in samples)
    if workers == 1:
        for task in tasks:
            yield preprocess_worker(task)
        return
    # Applied before spawn, so native libraries in children initialize one thread.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    pending = deque()
    try:
        for _ in range(bound):
            task = next(tasks, None)
            if task is None:
                break
            pending.append(executor.submit(preprocess_worker, task))
        while pending:
            result = pending.popleft().result()
            yield result
            task = next(tasks, None)
            if task is not None:
                pending.append(executor.submit(preprocess_worker, task))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
