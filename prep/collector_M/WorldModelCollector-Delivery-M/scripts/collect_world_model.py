"""Entry point for a reward-free, goal-conditioned AirSim delivery dataset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import traceback
import uuid

from world_model_features import ACTION_NAMES, ACTION_SCALES, SLEW_LIMITS
from world_model_session import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "airsim-delivery-m.v1"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out", type=Path, default=ROOT / "data" / "runs")
    result.add_argument("--episodes", type=int, default=1)
    result.add_argument("--episode-seconds", type=float, default=180.0)
    result.add_argument("--steps", type=int, default=0, help="0 means simulation-time limit only")
    result.add_argument("--dt", type=float, default=0.2)
    result.add_argument("--altitude", type=float, default=30.0)
    result.add_argument("--min-altitude", type=float, default=10.0)
    result.add_argument("--max-altitude", type=float, default=60.0)
    result.add_argument("--seed", type=int, default=None)
    result.add_argument("--policy", choices=["goal", "explore"], default="goal")
    result.add_argument("--cruise-speed", type=float, default=3.0)
    result.add_argument("--noise-scale", type=float, default=0.35)
    result.add_argument("--noise-tau", type=float, default=2.0)
    group = result.add_mutually_exclusive_group()
    group.add_argument("--goal-ned", nargs=3, type=float, metavar=("X", "Y", "Z"))
    group.add_argument("--goal-offset", nargs=3, type=float, metavar=("DX", "DY", "DZ"))
    result.add_argument("--goal-min-distance", type=float, default=40.0)
    result.add_argument("--goal-max-distance", type=float, default=80.0)
    result.add_argument("--goal-height-jitter", type=float, default=8.0)
    result.add_argument("--goal-radius", type=float, default=6.0)
    result.add_argument("--goal-dwell", type=float, default=2.0)
    result.add_argument("--delivery-speed", type=float, default=0.5)
    result.add_argument("--stationary-speed", type=float, default=0.3)
    result.add_argument("--vehicle", default="Drone1")
    result.add_argument("--ip", default="127.0.0.1")
    result.add_argument("--port", type=int, default=41451)
    result.add_argument("--launch", action="store_true", help="start local CityEnviron if RPC is offline")
    result.add_argument("--exe", type=Path, default=ROOT.parent / "WindowsNoEditor" / "CityEnviron.exe")
    result.add_argument("--settings", type=Path, default=ROOT / "config" / "wm_settings.json")
    result.add_argument("--prepare-only", action="store_true",
                        help="write run metadata without connecting to or resetting AirSim")
    return result


def validate_args(args, argument_parser):
    numeric = [v for v in vars(args).values() if isinstance(v, float)]
    coordinates = (args.goal_ned or []) + (args.goal_offset or [])
    if not all(math.isfinite(v) for v in numeric + coordinates):
        argument_parser.error("all numeric values must be finite")
    if args.episodes < 1 or args.steps < 0 or args.episode_seconds <= 0:
        argument_parser.error("episodes/time must be positive and steps must be >= 0")
    if not 0.05 <= args.dt <= 1.0:
        argument_parser.error("dt must be between 0.05 and 1 simulation seconds")
    if not 0 < args.min_altitude < args.max_altitude:
        argument_parser.error("require 0 < min-altitude < max-altitude")
    if not args.min_altitude + 2 <= args.altitude <= args.max_altitude - 2:
        argument_parser.error("initial altitude needs 2m margin inside altitude bounds")
    if not 0 < args.cruise_speed <= 4:
        argument_parser.error("cruise-speed must be in (0, 4]")
    if args.noise_scale < 0 or args.noise_tau <= 0:
        argument_parser.error("noise-scale must be >= 0 and noise-tau must be > 0")
    if not 0 < args.goal_min_distance <= args.goal_max_distance:
        argument_parser.error("require 0 < goal-min-distance <= goal-max-distance")
    if min(args.goal_radius, args.goal_dwell, args.delivery_speed, args.stationary_speed) <= 0:
        argument_parser.error("goal radius/dwell and speed thresholds must be positive")
    if args.goal_height_jitter < 0 or not 1 <= args.port <= 65535:
        argument_parser.error("invalid height jitter or RPC port")
    if args.goal_ned is not None and not args.min_altitude <= -args.goal_ned[2] <= args.max_altitude:
        argument_parser.error("absolute goal altitude is outside configured bounds")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rpc_online(ip, port):
    try:
        with socket.create_connection((ip, port), timeout=1):
            return True
    except OSError:
        return False


def connect(args):
    import airsim
    launched = False
    if not rpc_online(args.ip, args.port):
        if not args.launch:
            raise ConnectionError("AirSim is offline. Start CityEnviron or pass --launch.")
        if args.ip not in ("127.0.0.1", "localhost"):
            raise ValueError("--launch requires a local IP")
        if not args.exe.is_file():
            raise FileNotFoundError(str(args.exe))
        subprocess.Popen([str(args.exe.resolve()), "-windowed", "-ResX=960", "-ResY=540",
                          '-settings=' + str(args.settings.resolve())],
                         cwd=str(args.exe.parent))
        launched = True
        print("Waiting for CityEnviron RPC...", flush=True)
        deadline = time.monotonic() + 300
        while not rpc_online(args.ip, args.port):
            if time.monotonic() > deadline:
                raise TimeoutError("AirSim RPC did not start within 5 minutes")
            time.sleep(1)
    client = airsim.MultirotorClient(ip=args.ip, port=args.port, timeout_value=30)
    if not client.ping():
        raise ConnectionError("RPC port is open but AirSim ping failed")
    if args.vehicle not in client.listVehicles():
        raise ValueError("vehicle %s is missing. Restart CityEnviron with this branch's settings." %
                         args.vehicle)
    try:
        runtime_settings = json.loads(client.getSettingsString())
        runtime_settings_error = None
    except Exception as exc:
        runtime_settings = None
        runtime_settings_error = str(exc)
    return {"launched_by_this_run": launched, "runtime_settings": runtime_settings,
            "runtime_settings_error": runtime_settings_error}


def config_dict(args):
    keys = ["episode_seconds", "steps", "dt", "altitude", "min_altitude", "max_altitude",
            "policy", "cruise_speed", "noise_scale", "noise_tau", "goal_ned", "goal_offset",
            "goal_min_distance", "goal_max_distance", "goal_height_jitter", "goal_radius",
            "goal_dwell", "delivery_speed", "stationary_speed", "vehicle", "ip", "port"]
    return {key: getattr(args, key) for key in keys}


def run_episode(run_dir, number, args, metadata):
    episode_dir = run_dir / ("episode_%04d" % number)
    session = episode_dir / ".ipc"
    session.mkdir(parents=True)
    seed = metadata["seed"] + number * 1009
    episode_metadata = {
        "schema": SCHEMA, "episode_index": number, "status": "running",
        "config": config_dict(args),
        "settings": metadata["connection"]["runtime_settings"] or metadata["settings"],
        "settings_source": ("server_runtime" if metadata["connection"]["runtime_settings"]
                            else "local_requested_unverified"),
        "seeds": {"goal": seed + 11, "policy": seed + 23, "initial_yaw": seed + 37},
    }
    write_json(episode_dir / "episode.json", episode_metadata)
    processes = []
    logs = []
    try:
        for name, script in (("record", "world_model_record.py"),
                             ("control", "world_model_path.py")):
            log = (episode_dir / (name + ".log")).open("w", encoding="utf-8")
            logs.append(log)
            process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "scripts" / script),
                 "--episode-dir", str(episode_dir)],
                stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
            processes.append(process)
        last_message = time.monotonic()
        while any(p.poll() is None for p in processes):
            error = read_json(session / "error.json")
            if error:
                raise RuntimeError("%s process failed: %s" % (error["process"], error["message"]))
            for process in processes:
                if process.poll() not in (None, 0):
                    raise RuntimeError("worker exited with code %s. See episode logs." % process.returncode)
            if time.monotonic() - last_message > 30:
                requests = len(list(session.glob("ack_*.json")))
                print("Episode %d, committed snapshots %d" % (number, requests), flush=True)
                last_message = time.monotonic()
            time.sleep(0.1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("a worker failed. See episode logs.")
        result = read_json(episode_dir / "result.json")
        if not result or result["status"] != "complete":
            raise RuntimeError("workers stopped without a complete episode result")
        result["cleanup"] = read_json(session / "cleanup.json")
        episode_metadata["status"] = "complete"
        write_json(episode_dir / "result.json", result)
        write_json(episode_dir / "episode.json", episode_metadata)
        print("Episode %d complete: %s, steps=%d, sim=%.3fs" %
              (number, ",".join(result["end_reasons"]), result["transitions"],
               result["elapsed_sim_s"]), flush=True)
        return result
    except BaseException as exc:
        (session / "STOP").touch()
        episode_metadata.update(status="incomplete", error_type=type(exc).__name__, error=str(exc))
        write_json(episode_dir / "episode.json", episode_metadata)
        deadline = time.monotonic() + 10
        while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
            time.sleep(0.1)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait(timeout=10)
        raise
    finally:
        for log in logs:
            log.close()


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    validate_args(args, argument_parser)
    settings = json.loads(args.settings.read_text(encoding="utf-8-sig"))
    if args.vehicle not in settings.get("Vehicles", {}):
        argument_parser.error("vehicle is missing from settings file")
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    run_dir = args.out.resolve() / (datetime.now().strftime("run_%Y%m%d_%H%M%S_") +
                                    uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    versions = {}
    for package in ("airsim", "numpy", "msgpack", "msgpack-rpc-python", "tornado"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {
        "schema": SCHEMA, "status": "prepared",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "config": config_dict(args),
        "settings_path": str(args.settings.resolve()), "settings": settings,
        "settings_sha256": sha256(args.settings),
        "code_sha256": {p.name: sha256(p) for p in sorted((ROOT / "scripts").glob("*.py"))},
        "branch": read_json(ROOT / "BRANCH.json"),
        "python": sys.version, "python_executable": sys.executable,
        "dependencies": versions, "command_line": sys.argv,
        "scene": {"name": "CityEnviron", "exe_path": str(args.exe.resolve()),
                  "exe_sha256": sha256(args.exe) if args.exe.is_file() else None,
                  "weather_parameters": None, "wind_ned_m_s": None},
        "action_spec": {"names": ACTION_NAMES, "normalization_scales": ACTION_SCALES,
                        "slew_limits_per_s": SLEW_LIMITS,
                        "semantics": "velocity setpoints, heading frozen at interval start"},
        "coordinate_system": {"world": "AirSim local NED in meters; z positive down",
                              "heading": "yaw-only forward/right/down",
                              "body": "full quaternion forward/right/down",
                              "quaternion": "xyzw",
                              "angular_kinematics_frame": "AirSim API native; no frame conversion asserted"},
        "reward_storage": "deferred_from_measurements_and_events",
        "calibration": "runtime FOV/projection and actual image poses are saved per episode/snapshot",
    }
    write_json(run_dir / "run.json", metadata)
    print("Output: " + str(run_dir), flush=True)
    print("Seed: %d" % seed, flush=True)
    if args.prepare_only:
        return
    try:
        metadata["connection"] = connect(args)
        metadata["status"] = "running"
        write_json(run_dir / "run.json", metadata)
        results = [run_episode(run_dir, number, args, metadata)
                   for number in range(args.episodes)]
        metadata["status"] = "complete"
        metadata["results"] = results
        write_json(run_dir / "run.json", metadata)
    except BaseException as exc:
        metadata.update(status="incomplete", error_type=type(exc).__name__, error=str(exc))
        write_json(run_dir / "run.json", metadata)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

