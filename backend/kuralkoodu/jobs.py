"""Cancelable background jobs with progress + logs."""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from .engines.base import Cancelled, EngineUnavailable


class Job:
    def __init__(self, kind: str, project_id: str | None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.project_id = project_id
        self.status = "queued"
        self.progress = 0.0
        self.message = ""
        self.logs: list[str] = []
        self.result = None
        self.error = None
        self.created = time.time()
        self._cancel = threading.Event()

    def log(self, msg):
        self.logs.append(f"{time.strftime('%H:%M:%S')} {msg}")
        self.message = msg

    def update(self, frac, msg=None):
        self.progress = max(self.progress, min(1.0, float(frac)))
        if msg and msg != self.message:
            self.log(msg)

    def cancelled(self):
        return self._cancel.is_set()

    def cancel(self):
        self._cancel.set()
        if self.status in ("queued", "running"):
            self.log("Cancel requested")

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "project_id": self.project_id, "status": self.status,
                "progress": round(self.progress, 3), "message": self.message, "logs": self.logs[-200:],
                "result": self.result, "error": self.error}


MAX_KEPT_JOBS = 200


class JobManager:
    def __init__(self, workers=2):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.jobs: dict[str, Job] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _project_lock(self, project_id):
        with self._guard:
            return self._locks.setdefault(project_id or "-", threading.Lock())

    def _prune(self):
        finished = [j for j in self.jobs.values() if j.status in ("done", "error", "cancelled")]
        for j in sorted(finished, key=lambda j: j.created)[: max(0, len(self.jobs) - MAX_KEPT_JOBS)]:
            self.jobs.pop(j.id, None)

    def submit(self, kind, project_id, fn, *args, **kwargs) -> Job:
        self._prune()
        job = Job(kind, project_id)
        self.jobs[job.id] = job
        lock = self._project_lock(project_id)

        def run():
            # one job at a time per song: jobs share project.json, the separation and the render folders
            if not lock.acquire(blocking=False):
                job.message = "Waiting for the other job on this song to finish"
                lock.acquire()
            try:
                _run()
            finally:
                lock.release()

        def _run():
            if job.cancelled():
                job.status = "cancelled"
                return
            job.status = "running"
            job.log(f"{kind} started")
            try:
                # results go to JSON: convert numpy scalars once, here
                job.result = json.loads(json.dumps(fn(job, *args, **kwargs), default=float))
                job.status = "done"
                job.progress = 1.0
                job.log(f"{kind} finished")
            except Cancelled:
                job.status = "cancelled"
                job.log("Cancelled")
            except EngineUnavailable as e:
                job.status = "error"
                job.error = str(e)
                job.log(f"Engine unavailable: {e}")
            except Exception as e:  # never fail silently
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.log(traceback.format_exc()[-1500:])

        self.pool.submit(run)
        return job

    def get(self, job_id) -> Job | None:
        return self.jobs.get(job_id)
