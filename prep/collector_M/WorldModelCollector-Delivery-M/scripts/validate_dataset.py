"""Validate episode alignment, physics measurements, assets and reward absence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

from world_model_features import (ACTION_SCALES, SLEW_LIMITS, collision_event,
                                  goal_features, heading_to_world, norm, update_dwell)
from world_model_session import read_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, label, tolerance=1e-6):
    require(math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance),
            "%s mismatch: %r vs %r" % (label, a, b))


def no_scalar_reward(value):
    if isinstance(value, dict):
        require("reward" not in value and "rewards" not in value, "scalar reward field found")
        for child in value.values():
            no_scalar_reward(child)
    elif isinstance(value, list):
        for child in value:
            no_scalar_reward(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite JSON number")


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_assets(episode, observation):
    expected = {"front_rgb", "down_rgb", "front_depth", "down_depth"}
    require(set(observation["images"]) == expected, "missing image channels")
    for key, image in observation["images"].items():
        path = episode / image["path"]
        require(path.is_file(), "missing asset " + str(path))
        h, w = image["height"], image["width"]
        if key.endswith("depth"):
            depth = np.load(str(path), allow_pickle=False)
            mask = np.load(str(episode / image["valid_mask_path"]), allow_pickle=False)
            require(depth.dtype == np.float32 and depth.shape == (h, w), "depth dtype/shape mismatch")
            require(mask.dtype == np.uint8 and mask.shape == depth.shape, "mask dtype/shape mismatch")
            require(np.array_equal(mask, (np.isfinite(depth) & (depth > 0)).astype(np.uint8)),
                    "invalid depth mask")
            close(image["summary"]["valid_fraction"], float(mask.mean()), "valid depth fraction")
        else:
            with path.open("rb") as stream:
                header = stream.read(24)
            require(header[:8] == b"\x89PNG\r\n\x1a\n", "invalid PNG")
            require(struct.unpack(">II", header[16:24]) == (w, h), "PNG dimensions mismatch")


def validate_episode(episode, metadata_only=False):
    metadata = read_json(episode / "episode.json")
    require(metadata is not None and metadata["schema"] == "airsim-delivery-m.v1", "unknown episode schema")
    require(metadata["status"] == "complete", "episode is incomplete")
    task = read_json(episode / "task.json")
    result = read_json(episode / "result.json")
    observations = read_lines(episode / "observations.jsonl")
    transitions = read_lines(episode / "transitions.jsonl")
    require(len(transitions) > 0, "episode contains no transitions")
    require(len(observations) == len(transitions) + 1, "N+1 observation invariant failed")
    require(result["transitions"] == len(transitions) and result["observations"] == len(observations),
            "result counts mismatch")
    config = metadata["config"]
    origin = observations[0]["state"]["timestamp_ns"]
    for index, observation in enumerate(observations):
        no_scalar_reward(observation)
        require(observation["observation_id"] == index, "observation index mismatch")
        timing = observation["timing"]
        stamp = observation["state"]["timestamp_ns"]
        require(timing["state_before_images_ns"] == timing["state_after_images_ns"] == stamp,
                "simulation advanced during capture")
        close(observation["elapsed_sim_s"], (stamp - origin) / 1e9, "elapsed time")
        recomputed = goal_features(observation["state"], task["goal"],
                                   observation["goal"]["dwell_sim_s"])
        for name in ("distance_3d_m", "distance_horizontal_m", "height_error_up_m",
                     "bearing_error_rad", "closing_speed_m_s"):
            close(observation["goal"][name], recomputed[name], name)
        for name in ("offset_ned_m", "offset_heading_frd_m", "offset_body_frd_m"):
            for actual, expected in zip(observation["goal"][name], recomputed[name]):
                close(actual, expected, name)
        if not metadata_only:
            check_assets(episode, observation)
    stationary_s = 0.0
    cumulative_path = 0.0
    cumulative_progress = 0.0
    for index, transition in enumerate(transitions):
        no_scalar_reward(transition)
        before, after = observations[index:index + 2]
        require(transition["transition_id"] == transition["observation_id"] == index,
                "transition index mismatch")
        require(transition["next_observation_id"] == index + 1, "next observation mismatch")
        dt = (after["state"]["timestamp_ns"] - before["state"]["timestamp_ns"]) / 1e9
        require(dt > 0, "nonpositive simulation dt")
        close(transition["timing"]["actual_step_sim_s"], dt, "actual step time")
        require(transition["timing"]["command_queued_at_sim_ns"] == before["state"]["timestamp_ns"],
                "action start is not aligned with observation")
        require(transition["timing"]["state_after_advance_ns"] == after["state"]["timestamp_ns"],
                "action end is not aligned with next observation")
        action = transition["action"]
        require(len(action["applied"]) == 4, "expected 4D applied action")
        previous_applied = transitions[index - 1]["action"]["applied"] if index else [0.0] * 4
        for i, scale in enumerate(ACTION_SCALES):
            close(action["normalization_scales"][i], scale, "action scale")
            close(action["previous_applied"][i], previous_applied[i], "previous action")
            close(action["normalized_applied"][i], action["applied"][i] / scale, "action normalization")
            require(abs(action["applied"][i]) <= scale + 1e-6, "action bound exceeded")
            require(abs(action["applied"][i] - previous_applied[i]) <=
                    SLEW_LIMITS[i] * config["dt"] + 1e-6, "action slew exceeded")
        world = heading_to_world(action["applied"][:3], before["state"]["yaw_rad"])
        for actual, expected in zip(action["api"]["velocity_ned_m_s"], world):
            close(actual, expected, "API velocity")
        close(action["api"]["yaw_or_rate_deg_s"], action["applied"][3], "API yaw rate")
        m = transition["measurements"]
        progress = before["goal"]["distance_3d_m"] - after["goal"]["distance_3d_m"]
        close(m["goal_progress_3d_m"], progress, "goal progress")
        close(m["goal_progress_rate_m_s"], progress / dt, "goal progress rate")
        displacement = norm([after["state"]["position_ned_m"][i] -
                             before["state"]["position_ned_m"][i] for i in range(3)])
        close(m["displacement_3d_m"], displacement, "displacement")
        cumulative_path += displacement
        cumulative_progress += progress
        close(m["cumulative_sampled_path_length_m"], cumulative_path, "cumulative path")
        close(m["cumulative_goal_progress_m"], cumulative_progress, "cumulative progress")
        dwell = update_dwell(before["goal"], after["goal"], before["goal"]["dwell_sim_s"], dt)
        close(after["goal"]["dwell_sim_s"], dwell, "goal dwell")
        stationary = (before["goal"]["speed_3d_m_s"] <= config["stationary_speed"] and
                      after["goal"]["speed_3d_m_s"] <= config["stationary_speed"])
        stationary_s = stationary_s + dt if stationary else 0.0
        close(m["stationary_duration_sim_s"], stationary_s, "stationary duration")
        collision = collision_event(before["collision"], after["collision"])
        success = (after["goal"]["inside_goal"] and
                   after["goal"]["speed_below_delivery_threshold"] and
                   dwell >= task["goal"]["dwell_required_sim_s"])
        require(transition["events"]["collision"] == collision, "collision event mismatch")
        require(transition["events"]["delivery_success"] == success, "success event mismatch")
        require(transition["terminated"] == (collision or success), "terminated flag mismatch")
        limit = after["elapsed_sim_s"] >= config["episode_seconds"] or (
            config["steps"] > 0 and index + 1 >= config["steps"])
        require(transition["truncated"] == (limit and not transition["terminated"]), "truncated mismatch")
        require(transition["done"] == (transition["terminated"] or transition["truncated"]),
                "done mismatch")
        require(transition["done"] == (index == len(transitions) - 1),
                "terminal boundary appears in middle of episode")
    require(result["end_reasons"] == transitions[-1]["end_reasons"], "result ending mismatch")
    require(result["terminated"] == transitions[-1]["terminated"] and
            result["truncated"] == transitions[-1]["truncated"], "result flags mismatch")
    print("PASS %s: %d transitions, %d snapshots, %.6fs simulation, max dt error %.6fs" %
          (episode, len(transitions), len(observations), observations[-1]["elapsed_sim_s"],
           max(abs(t["timing"]["step_error_sim_s"]) for t in transitions)))
    return len(transitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a run directory or one episode directory")
    parser.add_argument("--metadata-only", action="store_true", help="skip image/depth file validation")
    args = parser.parse_args()
    try:
        if (args.path / "episode.json").is_file():
            episodes = [args.path]
        else:
            require((args.path / "run.json").is_file(), "path must be a run or episode directory")
            require(read_json(args.path / "run.json")["status"] == "complete", "run is incomplete")
            episodes = sorted(args.path.glob("episode_*"))
        require(bool(episodes), "no episodes found")
        total = sum(validate_episode(episode, args.metadata_only) for episode in episodes)
        print("Validated %d episodes / %d transitions" % (len(episodes), total))
        return 0
    except Exception as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

