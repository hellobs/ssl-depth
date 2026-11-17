"""AirSim snapshots and lossless image/depth assets."""
from __future__ import annotations

import math
import time
from pathlib import Path

import airsim
import numpy as np

from world_model_features import goal_features, yaw_from_quaternion


IMAGE_SPECS = [
    ("front_rgb", "front_center", airsim.ImageType.Scene, False),
    ("down_rgb", "bottom_center", airsim.ImageType.Scene, False),
    ("front_depth", "front_center", airsim.ImageType.DepthPlanar, True),
    ("down_depth", "bottom_center", airsim.ImageType.DepthPlanar, True),
]


def vector(value):
    return [float(value.x_val), float(value.y_val), float(value.z_val)]


def quaternion(value):
    return [float(value.x_val), float(value.y_val), float(value.z_val), float(value.w_val)]


def pose(value):
    return {"position_ned_m": vector(value.position),
            "orientation_xyzw": quaternion(value.orientation)}


def state_dict(state):
    k = state.kinematics_estimated
    q = quaternion(k.orientation)
    return {
        "timestamp_ns": int(state.timestamp),
        "position_ned_m": vector(k.position),
        "orientation_xyzw": q,
        "yaw_rad": yaw_from_quaternion(q),
        "linear_velocity_ned_m_s": vector(k.linear_velocity),
        "angular_velocity_api_rad_s": vector(k.angular_velocity),
        "linear_acceleration_ned_m_s2": vector(k.linear_acceleration),
        "angular_acceleration_api_rad_s2": vector(k.angular_acceleration),
        "landed_state": int(state.landed_state),
        "ready": bool(state.ready),
    }


def collision_dict(info):
    return {
        "has_collided": bool(info.has_collided),
        "timestamp_ns": int(info.time_stamp),
        "object_name": str(info.object_name),
        "object_id": int(info.object_id),
        "normal_api": vector(info.normal),
        "impact_point_ned_m": vector(info.impact_point),
        "position_ned_m": vector(info.position),
        "penetration_depth_m": float(info.penetration_depth),
    }


def calibrations(client, vehicle, settings):
    result = {}
    captures = settings.get("CameraDefaults", {}).get("CaptureSettings", [])
    rgb = next((c for c in captures if c["ImageType"] == 0), None)
    depth = next((c for c in captures if c["ImageType"] == 1), None)
    mounts = settings.get("Vehicles", {}).get(vehicle, {}).get("Cameras", {})
    for name in ("front_center", "bottom_center"):
        info = client.simGetCameraInfo(name, vehicle_name=vehicle)
        result[name] = {
            "runtime_horizontal_fov_deg": float(info.fov),
            "runtime_projection_matrix": info.proj_mat.matrix,
            "runtime_pose_at_initialization": pose(info.pose),
            "nominal_mount_from_settings": mounts.get(name),
            "capture_settings": {"rgb": rgb, "depth_planar": depth},
            "pose_convention": "AirSim world NED, quaternion xyzw",
        }
    return result


def depth_summary(depth, valid):
    values = depth[valid]
    summary = {"valid_fraction": float(valid.mean()), "min_m": None, "median_m": None,
               "p10_m": None, "center_p10_m": None}
    if values.size:
        summary.update(min_m=float(values.min()), median_m=float(np.median(values)),
                       p10_m=float(np.percentile(values, 10)))
    h, w = depth.shape
    region = depth[h // 3:2 * h // 3, w // 3:2 * w // 3]
    mask = valid[h // 3:2 * h // 3, w // 3:2 * w // 3]
    if mask.any():
        summary["center_p10_m"] = float(np.percentile(region[mask], 10))
    return summary


def capture(client, episode_dir, index, goal, vehicle, camera_metadata):
    if not client.simIsPause():
        raise RuntimeError("recorder requires the simulation to be paused")
    wall_start = time.monotonic()
    before = client.getMultirotorState(vehicle_name=vehicle)
    requests = [airsim.ImageRequest(camera, kind, is_float, not is_float)
                for _, camera, kind, is_float in IMAGE_SPECS]
    responses = client.simGetImages(requests, vehicle_name=vehicle)
    if len(responses) != len(IMAGE_SPECS):
        raise RuntimeError("incomplete image response")
    collision = collision_dict(client.simGetCollisionInfo(vehicle_name=vehicle))
    environment = client.simGetGroundTruthEnvironment(vehicle_name=vehicle)
    after = client.getMultirotorState(vehicle_name=vehicle)
    if before.timestamp != after.timestamp or not client.simIsPause():
        raise RuntimeError("simulation advanced during snapshot capture")
    image_dir = Path(episode_dir) / "observations"
    image_dir.mkdir(parents=True, exist_ok=True)
    images = {}
    for spec, response in zip(IMAGE_SPECS, responses):
        key, camera, image_type, is_float = spec
        width, height = int(response.width), int(response.height)
        if width <= 0 or height <= 0:
            raise RuntimeError("empty camera response for " + key + ": " + response.message)
        stem = "%06d_%s" % (index, key)
        fov = camera_metadata[camera]["runtime_horizontal_fov_deg"]
        focal = width / (2 * math.tan(math.radians(fov) / 2))
        entry = {
            "camera": camera, "image_type": int(image_type),
            "timestamp_ns": int(response.time_stamp),
            "width": width, "height": height,
            "camera_world_pose": {
                "position_ned_m": vector(response.camera_position),
                "orientation_xyzw": quaternion(response.camera_orientation),
            },
            "intrinsics_K_estimated": [[focal, 0.0, width / 2.0],
                                       [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        }
        if is_float:
            depth = np.asarray(response.image_data_float, dtype=np.float32)
            if depth.size != width * height:
                raise RuntimeError("invalid depth buffer length for " + key)
            depth = depth.reshape(height, width)
            valid = np.isfinite(depth) & (depth > 0)
            filename = stem + ".npy"
            mask_file = stem + "_valid.npy"
            np.save(str(image_dir / filename), depth, allow_pickle=False)
            np.save(str(image_dir / mask_file), valid.astype(np.uint8), allow_pickle=False)
            entry.update(path="observations/" + filename, encoding="float32_npy",
                         units="m_camera_optical_axis_depth", valid_mask_path="observations/" + mask_file,
                         summary=depth_summary(depth, valid))
        else:
            filename = stem + ".png"
            data = bytes(response.image_data_uint8)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("RGB response is not a PNG for " + key)
            (image_dir / filename).write_bytes(data)
            entry.update(path="observations/" + filename, encoding="png_rgb")
        images[key] = entry
    state = state_dict(after)
    stamps = [value["timestamp_ns"] for value in images.values()]
    return {
        "observation_id": index,
        "state": state,
        "goal": goal_features(state, goal, 0.0),
        "collision": collision,
        "environment": {
            "position_ned_m": vector(environment.position),
            "gravity_ned_m_s2": vector(environment.gravity),
            "air_pressure_pa": float(environment.air_pressure),
            "temperature_k": float(environment.temperature),
            "air_density_kg_m3": float(environment.air_density),
            "geo_point": {"latitude": float(environment.geo_point.latitude),
                          "longitude": float(environment.geo_point.longitude),
                          "altitude_m": float(environment.geo_point.altitude)},
        },
        "images": images,
        "timing": {
            "state_before_images_ns": int(before.timestamp),
            "state_after_images_ns": int(after.timestamp),
            "image_timestamp_spread_s": (max(stamps) - min(stamps)) / 1e9,
            "image_minus_state_s": {key: (value["timestamp_ns"] - int(after.timestamp)) / 1e9
                                    for key, value in images.items()},
            "capture_and_save_wall_s": time.monotonic() - wall_start,
        },
    }

