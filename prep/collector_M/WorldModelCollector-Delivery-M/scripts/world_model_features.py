"""Pure geometry, behavior policy and reward-free delivery measurements."""
from __future__ import annotations

import math
import random


ACTION_NAMES = ["forward_m_s", "right_m_s", "down_m_s", "yaw_rate_deg_s"]
ACTION_SCALES = [4.0, 3.0, 2.0, 30.0]
SLEW_LIMITS = [2.0, 2.0, 1.0, 40.0]


def norm(vector):
    return math.sqrt(sum(float(v) ** 2 for v in vector))


def difference(a, b):
    return [float(x) - float(y) for x, y in zip(a, b)]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def wrap_radians(value):
    return math.atan2(math.sin(value), math.cos(value))


def yaw_from_quaternion(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def world_to_heading(vector, yaw):
    x, y, z = vector
    c, s = math.cos(yaw), math.sin(yaw)
    return [c * x + s * y, -s * x + c * y, z]


def heading_to_world(vector, yaw):
    forward, right, down = vector
    c, s = math.cos(yaw), math.sin(yaw)
    return [c * forward - s * right, s * forward + c * right, down]


def world_to_body(vector, q):
    # Inverse quaternion rotation, q maps body FRD to world NED.
    x, y, z, w = q
    rotation = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]
    return [sum(rotation[j][i] * vector[j] for j in range(3)) for i in range(3)]


def goal_features(state, goal, dwell_s):
    offset = difference(goal["position_ned_m"], state["position_ned_m"])
    horizontal = norm(offset[:2])
    distance = norm(offset)
    yaw = state["yaw_rad"]
    bearing_valid = horizontal > 1e-6
    bearing = wrap_radians(math.atan2(offset[1], offset[0]) - yaw) if bearing_valid else 0.0
    speed = norm(state["linear_velocity_ned_m_s"])
    closing = sum(offset[i] * state["linear_velocity_ned_m_s"][i]
                  for i in range(3)) / distance if distance > 1e-6 else 0.0
    return {
        "offset_ned_m": offset,
        "offset_heading_frd_m": world_to_heading(offset, yaw),
        "offset_body_frd_m": world_to_body(offset, state["orientation_xyzw"]),
        "distance_3d_m": distance,
        "distance_horizontal_m": horizontal,
        "height_error_up_m": -offset[2],
        "bearing_error_rad": bearing,
        "bearing_error_sin": math.sin(bearing),
        "bearing_error_cos": math.cos(bearing),
        "bearing_valid": bearing_valid,
        "closing_speed_m_s": closing,
        "inside_goal": distance <= goal["radius_m"],
        "speed_3d_m_s": speed,
        "speed_below_delivery_threshold": speed <= goal["max_delivery_speed_m_s"],
        "dwell_sim_s": dwell_s,
    }


def update_dwell(previous_goal, current_goal, previous_dwell, dt):
    previous_ok = (previous_goal["inside_goal"] and
                   previous_goal["speed_below_delivery_threshold"])
    current_ok = (current_goal["inside_goal"] and
                  current_goal["speed_below_delivery_threshold"])
    if not current_ok:
        return 0.0
    return previous_dwell + dt if previous_ok else 0.0


def collision_event(previous, current):
    # AirSim collision state can remain latched after a past contact.
    return bool(current["has_collided"] and
                (not previous["has_collided"] or
                 current["timestamp_ns"] != previous["timestamp_ns"]))


def sample_goal(initial_position, config, seed):
    if config["goal_ned"] is not None:
        position = list(config["goal_ned"])
        method = "explicit_absolute_ned"
    elif config["goal_offset"] is not None:
        position = [initial_position[i] + config["goal_offset"][i] for i in range(3)]
        method = "explicit_offset_ned"
    else:
        rng = random.Random(seed)
        angle = rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(config["goal_min_distance"], config["goal_max_distance"])
        altitude = clamp(-initial_position[2] + rng.uniform(-config["goal_height_jitter"],
                                                           config["goal_height_jitter"]),
                         config["min_altitude"], config["max_altitude"])
        position = [initial_position[0] + radius * math.cos(angle),
                    initial_position[1] + radius * math.sin(angle), -altitude]
        method = "seeded_random_annulus_with_height_jitter"
    if not config["min_altitude"] <= -position[2] <= config["max_altitude"]:
        raise ValueError("goal altitude is outside the configured altitude envelope")
    return {
        "position_ned_m": position,
        "radius_m": config["goal_radius"],
        "dwell_required_sim_s": config["goal_dwell"],
        "max_delivery_speed_m_s": config["delivery_speed"],
        "sampling_method": method,
        "seed": seed,
        "initial_distance_3d_m": norm(difference(position, initial_position)),
        "reachability_checked": False,
    }


class BehaviorPolicy:
    """Goal tracking plus correlated exploration, with explicit applied commands."""

    def __init__(self, config, seed):
        self.config = config
        self.rng = random.Random(seed)
        self.noise = [0.0] * 4

    def command(self, observation, previous_applied):
        config = self.config
        dt = config["dt"]
        alpha = math.exp(-dt / config["noise_tau"])
        sigma = [0.7, 0.7, 0.25, 5.0]
        for i in range(4):
            self.noise[i] = (alpha * self.noise[i] +
                             math.sqrt(1 - alpha * alpha) * sigma[i] *
                             self.rng.gauss(0.0, 1.0))
        goal = observation["goal"]
        offset = goal["offset_heading_frd_m"]
        # Noise fades near the destination so intentional delivery dwell is possible.
        fade = clamp(goal["distance_3d_m"] / 20.0, 0.0, 1.0)
        if config["policy"] == "explore":
            requested = [3 * self.noise[0], 3 * self.noise[1],
                         self.noise[2], 3 * self.noise[3]]
        else:
            requested = [0.4 * offset[0], 0.4 * offset[1], 0.4 * offset[2],
                         math.degrees(goal["bearing_error_rad"]) * 0.8 * fade]
            horizontal_speed = norm(requested[:2])
            if horizontal_speed > config["cruise_speed"]:
                requested[:2] = [v * config["cruise_speed"] / horizontal_speed
                                 for v in requested[:2]]
            requested = [requested[i] + config["noise_scale"] * fade * self.noise[i]
                         for i in range(4)]
        bounded = [clamp(v, -limit, limit) for v, limit in zip(requested, ACTION_SCALES)]
        altitude = -observation["state"]["position_ned_m"][2]
        guards = []
        if altitude < config["min_altitude"] + 2:
            bounded[2] = min(bounded[2], -min(1.0, config["min_altitude"] + 2 - altitude))
            guards.append("lower_altitude_recovery")
        if altitude > config["max_altitude"] - 2:
            bounded[2] = max(bounded[2], min(1.0, altitude - config["max_altitude"] + 2))
            guards.append("upper_altitude_recovery")
        applied = [clamp(bounded[i], previous_applied[i] - SLEW_LIMITS[i] * dt,
                         previous_applied[i] + SLEW_LIMITS[i] * dt) for i in range(4)]
        world = heading_to_world(applied[:3], observation["state"]["yaw_rad"])
        # Longer task lifetime keeps the command active across frame quantization.
        # The next step replaces it while the simulator is paused.
        duration = dt + 1.0
        return {
            "names": ACTION_NAMES,
            "frame": "yaw_aligned_horizontal_FRD_with_world_NED_down",
            "requested": requested,
            "bounded": bounded,
            "applied": applied,
            "normalized_applied": [v / scale for v, scale in zip(applied, ACTION_SCALES)],
            "normalization_scales": ACTION_SCALES,
            "previous_applied": list(previous_applied),
            "slew_limits_per_s": SLEW_LIMITS,
            "altitude_guards": guards,
            "exploration_noise": list(self.noise),
            "api": {
                "method": "moveByVelocityAsync",
                "velocity_ned_m_s": world,
                "duration_s": duration,
                "drivetrain": "MaxDegreeOfFreedom",
                "yaw_mode_is_rate": True,
                "yaw_or_rate_deg_s": applied[3],
                "vehicle_name": config["vehicle"],
            },
        }


def transition_measurements(before, after, command, dt, stationary_s):
    position_delta = difference(after["state"]["position_ned_m"],
                                before["state"]["position_ned_m"])
    distance = norm(position_delta)
    progress = before["goal"]["distance_3d_m"] - after["goal"]["distance_3d_m"]
    action_delta = difference(command["applied"], command["previous_applied"])
    return {
        "time_alive_increment_sim_s": dt,
        "goal_progress_3d_m": progress,
        "goal_progress_horizontal_m": (before["goal"]["distance_horizontal_m"] -
                                       after["goal"]["distance_horizontal_m"]),
        "absolute_height_error_reduction_m": (abs(before["goal"]["height_error_up_m"]) -
                                              abs(after["goal"]["height_error_up_m"])),
        "goal_progress_rate_m_s": progress / dt,
        "displacement_ned_m": position_delta,
        "displacement_heading_frd_m": world_to_heading(position_delta,
                                                       before["state"]["yaw_rad"]),
        "displacement_3d_m": distance,
        "displacement_horizontal_m": norm(position_delta[:2]),
        "average_translation_speed_m_s": distance / dt,
        "speed_before_m_s": before["goal"]["speed_3d_m_s"],
        "speed_after_m_s": after["goal"]["speed_3d_m_s"],
        "vertical_speed_down_after_m_s": after["state"]["linear_velocity_ned_m_s"][2],
        "goal_dwell_before_sim_s": before["goal"]["dwell_sim_s"],
        "goal_dwell_after_sim_s": after["goal"]["dwell_sim_s"],
        "inside_goal_before": before["goal"]["inside_goal"],
        "inside_goal_after": after["goal"]["inside_goal"],
        "stationary_duration_sim_s": stationary_s,
        "action_delta": action_delta,
        "action_delta_l2_normalized": norm([v / scale for v, scale in
                                           zip(action_delta, ACTION_SCALES)]),
        "normalized_command_squared_integral": sum(v * v for v in
                                                   command["normalized_applied"]) * dt,
    }

