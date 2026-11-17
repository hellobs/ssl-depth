"""Recorder process. Owns observations, transition labels and episode endings."""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import airsim

from world_model_airsim import capture
from world_model_features import (collision_event, transition_measurements,
                                  update_dwell)
from world_model_session import append_json, read_json, wait_json, write_json


def run(episode_dir):
    episode_dir = Path(episode_dir)
    session = episode_dir / ".ipc"
    metadata = wait_json(episode_dir / "episode.json", session)
    config = metadata["config"]
    initialized = wait_json(session / "initialized.json", session)
    goal = initialized["goal"]
    camera_metadata = initialized["cameras"]
    client = airsim.MultirotorClient(ip=config["ip"], timeout_value=60,
                                    port=config["port"])
    client.ping()
    previous = None
    stationary_s = 0.0
    cumulative_path = 0.0
    cumulative_progress = 0.0
    index = 0
    with (episode_dir / "observations.jsonl").open("x", encoding="utf-8") as observations, \
            (episode_dir / "transitions.jsonl").open("x", encoding="utf-8") as transitions:
        while True:
            request = wait_json(session / ("request_%06d.json" % index), session)
            observation = capture(client, episode_dir, index, goal, config["vehicle"],
                                  camera_metadata)
            observation["elapsed_sim_s"] = (
                observation["state"]["timestamp_ns"] - initialized["initial_timestamp_ns"]) / 1e9
            stop = False
            result = None
            transition = None
            if previous is not None:
                dt = (observation["state"]["timestamp_ns"] -
                      previous["state"]["timestamp_ns"]) / 1e9
                if dt <= 0:
                    raise RuntimeError("nonpositive simulation interval")
                observation["goal"]["dwell_sim_s"] = update_dwell(
                    previous["goal"], observation["goal"],
                    previous["goal"]["dwell_sim_s"], dt)
                collision = collision_event(previous["collision"], observation["collision"])
                success = (observation["goal"]["inside_goal"] and
                           observation["goal"]["speed_below_delivery_threshold"] and
                           observation["goal"]["dwell_sim_s"] >= goal["dwell_required_sim_s"])
                stationary = (previous["goal"]["speed_3d_m_s"] <= config["stationary_speed"] and
                              observation["goal"]["speed_3d_m_s"] <= config["stationary_speed"])
                stationary_s = stationary_s + dt if stationary else 0.0
                measurements = transition_measurements(
                    previous, observation, request["action"], dt, stationary_s)
                cumulative_path += measurements["displacement_3d_m"]
                cumulative_progress += measurements["goal_progress_3d_m"]
                measurements["cumulative_sampled_path_length_m"] = cumulative_path
                measurements["cumulative_goal_progress_m"] = cumulative_progress
                time_limit = observation["elapsed_sim_s"] >= config["episode_seconds"]
                step_limit = config["steps"] > 0 and index >= config["steps"]
                terminated = collision or success
                truncated = (time_limit or step_limit) and not terminated
                reasons = []
                if collision:
                    reasons.append("collision")
                if success:
                    reasons.append("delivery_success")
                if time_limit and not terminated:
                    reasons.append("time_limit")
                if step_limit and not terminated:
                    reasons.append("step_limit")
                stop = terminated or truncated
                camera_skew = any(
                    abs(value) > 0.02
                    for snapshot in (previous, observation)
                    for value in snapshot["timing"]["image_minus_state_s"].values())
                transition = {
                    "transition_id": index - 1,
                    "observation_id": index - 1,
                    "next_observation_id": index,
                    "action": request["action"],
                    "timing": {
                        "nominal_step_sim_s": config["dt"],
                        "actual_step_sim_s": dt,
                        "step_error_sim_s": dt - config["dt"],
                        "command_queued_at_sim_ns": request["command_queued_at_sim_ns"],
                        "state_after_advance_ns": request["state_after_advance_ns"],
                        "advance_wall_s": request["advance_wall_s"],
                        "action_end_semantics": "replaced_or_cancelled_while_paused_after_snapshot",
                    },
                    "measurements": measurements,
                    "events": {
                        "collision": collision, "delivery_success": success,
                        "time_limit_reached": time_limit, "step_limit_reached": step_limit,
                    },
                    "terminated": terminated,
                    "truncated": truncated,
                    "done": stop,
                    "end_reasons": reasons,
                    "quality": {
                        "valid_for_dynamics": not camera_skew,
                        "step_error_exceeds_25pct": abs(dt - config["dt"]) > config["dt"] * 0.25,
                        "camera_state_skew_exceeds_20ms": camera_skew,
                    },
                }
                if stop:
                    result = {
                        "status": "complete",
                        "transitions": index,
                        "observations": index + 1,
                        "elapsed_sim_s": observation["elapsed_sim_s"],
                        "terminated": terminated, "truncated": truncated,
                        "end_reasons": reasons,
                        "delivery_success": success, "collision": collision,
                        "final_goal_distance_m": observation["goal"]["distance_3d_m"],
                        "cumulative_sampled_path_length_m": cumulative_path,
                    }
            append_json(observations, observation)
            if transition is not None:
                append_json(transitions, transition)
            if result is not None:
                write_json(episode_dir / "result.json", result)
            write_json(session / ("ack_%06d.json" % index),
                       {"observation": observation, "stop": stop, "result": result})
            if index % 25 == 0 or stop:
                print("snapshot=%d sim=%.3fs goal=%.2fm dwell=%.2fs" %
                      (index, observation["elapsed_sim_s"],
                       observation["goal"]["distance_3d_m"],
                       observation["goal"]["dwell_sim_s"]), flush=True)
            if stop:
                return
            previous = observation
            index += 1


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
                       {"process": "record", "type": type(exc).__name__, "message": str(exc)})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

