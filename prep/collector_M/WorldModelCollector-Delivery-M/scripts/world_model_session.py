"""Atomic, per-step coordination for the controller and recorder."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%d.tmp" % os.getpid())
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(str(temporary), str(path))


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def wait_json(path, session, timeout=120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        error = read_json(Path(session) / "error.json")
        if error:
            raise RuntimeError("%s: %s" % (error["process"], error["message"]))
        if (Path(session) / "STOP").exists():
            raise InterruptedError("collection interrupted")
        value = read_json(path)
        if value is not None:
            return value
        time.sleep(0.01)
    raise TimeoutError("timed out waiting for " + str(path))


def append_json(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()

