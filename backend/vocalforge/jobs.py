"""Cancelable background jobs with progress + logs."""
from __future__ import annotations

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


class JobManager:
    def __init__(self, workers=2):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.jobs: dict[str, Job] = {}

    def submit(self, kind, project_id, fn, *args, **kwargs) -> Job:
        job = Job(kind, project_id)
        self.jobs[job.id] = job

        def run():
            if job.cancelled():
                job.status = "cancelled"
                return
            job.status = "running"
            job.log(f"{kind} started")
            try:
                job.result = fn(job, *args, **kwargs)
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
