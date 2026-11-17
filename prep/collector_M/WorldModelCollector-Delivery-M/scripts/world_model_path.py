"""Controller process. Owns vehicle commands and the simulation pause/step clock."""
from __future__ import annotations

import argparse
import random
import time
import traceback
from pathlib import Path

import airsim

from world_model_airsim import calibrations, state_dict
from world_model_features import BehaviorPolicy, sample_goal
from world_model_session import read_json, wait_json, write_json


def advance(client, seconds, session):
    started = time.monotonic()
    client.simContinueForTime(seconds)
    while not client.simIsPause():
        if (session / "STOP").exists():
            raise InterruptedError("collection interrupted during advance")
        error = read_json(session / "error.json")
        if error:
            raise RuntimeError(error["message"])
        if time.monotonic() - started > 30:
            raise TimeoutError("simulation failed to pause after step")
        time.sleep(0.003)
    return time.monotonic() - started


def run(episode_dir):
    episode_dir = Path(episode_dir)
    session = episode_dir / ".ipc"
    metadata = wait_json(episode_dir / "episode.json", session)
    config = metadata["config"]
    vehicle = config["vehicle"]
    client = airsim.MultirotorClient(ip=config["ip"], port=config["port"], timeout_value=90)
    connected = False
    try:
        if not client.ping():
            raise RuntimeError("AirSim ping returned false")
        connected = True
        client.simPause(False)
        client.reset()
        client.enableApiControl(True, vehicle_name=vehicle)
        if not client.armDisarm(True, vehicle_name=vehicle):
            raise RuntimeError("arming failed")
        if not client.moveToZAsync(-config["altitude"], 3.0, timeout_sec=40,
                                  vehicle_name=vehicle).get():
            raise RuntimeError("initial altitude climb failed")
        initial_yaw = random.Random(metadata["seeds"]["initial_yaw"]).uniform(-180, 180)
        if not client.rotateToYawAsync(initial_yaw, timeout_sec=30, margin=3,
                                      vehicle_name=vehicle).get():
            raise RuntimeError("initial heading rotation failed")
        client.hoverAsync(vehicle_name=vehicle).get()
        time.sleep(0.4)
        client.simPause(True)
        initial = state_dict(client.getMultirotorState(vehicle_name=vehicle))
        goal = sample_goal(initial["position_ned_m"], config, metadata["seeds"]["goal"])
        cameras = calibrations(client, vehicle, metadata["settings"])
        write_json(session / "initialized.json",
                   {"goal": goal, "cameras": cameras,
                    "initial_timestamp_ns": initial["timestamp_ns"],
                    "initial_yaw_requested_deg": initial_yaw,
                    "initial_state": initial})
        write_json(episode_dir / "task.json",
                   {"goal": goal, "cameras": cameras, "initial_state": initial,
                    "initial_yaw_requested_deg": initial_yaw})
        write_json(session / "request_000000.json", {"kind": "initial"})
        ack = wait_json(session / "ack_000000.json", session)
        policy = BehaviorPolicy(config, metadata["seeds"]["policy"])
        previous_action = [0.0] * 4
        index = 1
        while not ack["stop"]:
            observation = ack["observation"]
            action = policy.command(observation, previous_action)
            api = action["api"]
            if not client.simIsPause():
                raise RuntimeError("simulation was resumed by another process")
            queued_at = int(client.getMultirotorState(vehicle_name=vehicle).timestamp)
            if queued_at != observation["state"]["timestamp_ns"]:
                raise RuntimeError("simulation state changed after recorder acknowledgement")
            velocity = api["velocity_ned_m_s"]
            pending = client.moveByVelocityAsync(
                velocity[0], velocity[1], velocity[2], api["duration_s"],
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(True, api["yaw_or_rate_deg_s"]),
                vehicle_name=vehicle)
            client.ping()  # Flush the async command onto the RPC connection before advancing.
            if pending.error is not None:
                raise RuntimeError(str(pending.error))
            advance_wall_s = advance(client, config["dt"], session)
            state_after = client.getMultirotorState(vehicle_name=vehicle)
            if pending.error is not None:
                raise RuntimeError(str(pending.error))
            if pending.result is not None:
                raise RuntimeError("velocity task ended before the next command replacement")
            write_json(session / ("request_%06d.json" % index),
                       {"kind": "transition", "action": action,
                        "command_queued_at_sim_ns": queued_at,
                        "state_after_advance_ns": int(state_after.timestamp),
                        "advance_wall_s": advance_wall_s})
            ack = wait_json(session / ("ack_%06d.json" % index), session)
            previous_action = action["applied"]
            index += 1
    finally:
        if connected:
            try:
                client.cancelLastTask(vehicle_name=vehicle)
                client.simPause(False)
                client.hoverAsync(vehicle_name=vehicle).get()
                write_json(session / "cleanup.json", {"hover_requested": True, "error": None})
            except Exception as exc:
                write_json(session / "cleanup.json",
                           {"hover_requested": False, "error": str(exc)})
                print("Cleanup warning: " + str(exc), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True)
    args = parser.parse_args()
    try:
        run(args.episode_dir)
    except BaseException as exc:
        error_path = Path(args.episode_dir) / ".ipc" / "error.json"
        if read_json(error_path) is None:
            write_json(error_path,
                       {"process": "control", "type": type(exc).__name__, "message": str(exc)})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

