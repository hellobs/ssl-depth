"""Targeted regressions for goal geometry, action alignment and event labels."""
from contextlib import redirect_stderr
import io
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_world_model import config_dict, parser, validate_args
from validate_dataset import no_scalar_reward
from world_model_features import (
    ACTION_SCALES, BehaviorPolicy, collision_event, goal_features, heading_to_world,
    sample_goal, transition_measurements, update_dwell, world_to_body,
    world_to_heading,
)
from world_model_session import read_json, wait_json, write_json


def state(position=None, yaw=0.0, velocity=None):
    return {
        "position_ned_m": position or [0.0, 0.0, -30.0],
        "yaw_rad": yaw,
        "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)],
        "linear_velocity_ned_m_s": velocity or [0.0, 0.0, 0.0],
    }


def goal():
    return {"position_ned_m": [10.0, 0.0, -35.0], "radius_m": 2.0,
            "max_delivery_speed_m_s": 0.5, "dwell_required_sim_s": 2.0}


def observation(s=None):
    s = s or state()
    return {"state": s, "goal": goal_features(s, goal(), 0.0)}


class GeometryTests(unittest.TestCase):
    def test_goal_signs(self):
        features = goal_features(state(yaw=math.pi / 2), goal(), 0.0)
        self.assertAlmostEqual(features["offset_heading_frd_m"][0], 0.0)
        self.assertAlmostEqual(features["offset_heading_frd_m"][1], -10.0)
        self.assertEqual(features["height_error_up_m"], 5.0)
        self.assertAlmostEqual(features["bearing_error_rad"], -math.pi / 2)
        self.assertAlmostEqual(features["distance_3d_m"], math.sqrt(125))

    def test_frame_round_trip(self):
        rng = random.Random(8)
        for _ in range(50):
            yaw = rng.uniform(-math.pi, math.pi)
            v = [rng.uniform(-100, 100) for _ in range(3)]
            recovered = heading_to_world(world_to_heading(v, yaw), yaw)
            for actual, expected in zip(recovered, v):
                self.assertAlmostEqual(actual, expected)

    def test_full_body_rotation_preserves_distance(self):
        q = [math.sin(0.3), 0.0, 0.0, math.cos(0.3)]
        result = world_to_body([1.0, 2.0, 3.0], q)
        self.assertAlmostEqual(sum(v * v for v in result), 14.0)
        self.assertNotAlmostEqual(result[2], 3.0)

    def test_zero_horizontal_bearing(self):
        f = goal_features(state([10.0, 0.0, -30.0]), goal(), 0.0)
        self.assertFalse(f["bearing_valid"])
        self.assertEqual(f["bearing_error_rad"], 0.0)

    def test_closing_speed(self):
        g = goal()
        g["position_ned_m"] = [10, 0, -30]
        f = goal_features(state(velocity=[2, 0, 0]), g, 0)
        self.assertEqual(f["closing_speed_m_s"], 2.0)


class TaskTests(unittest.TestCase):
    def test_dwell_requires_both_boundaries(self):
        good = {"inside_goal": True, "speed_below_delivery_threshold": True}
        outside = {"inside_goal": False, "speed_below_delivery_threshold": True}
        fast = {"inside_goal": True, "speed_below_delivery_threshold": False}
        self.assertAlmostEqual(update_dwell(good, good, 0.4, 0.2), 0.6)
        self.assertEqual(update_dwell(outside, good, 0.4, 0.2), 0.0)
        self.assertEqual(update_dwell(good, fast, 0.4, 0.2), 0.0)

    def test_collision_latch(self):
        old = {"has_collided": True, "timestamp_ns": 100}
        self.assertFalse(collision_event(old, dict(old)))
        self.assertTrue(collision_event(old, {"has_collided": True, "timestamp_ns": 101}))
        self.assertTrue(collision_event({"has_collided": False, "timestamp_ns": 0}, old))
        self.assertFalse(collision_event(old, {"has_collided": False, "timestamp_ns": 0}))

    def test_seeded_goal_and_altitude(self):
        cfg = config_dict(parser().parse_args([]))
        a = sample_goal([0, 0, -30], cfg, 123)
        b = sample_goal([0, 0, -30], cfg, 123)
        self.assertEqual(a, b)
        self.assertTrue(40 <= math.hypot(*a["position_ned_m"][:2]) <= 80)
        self.assertTrue(22 <= -a["position_ned_m"][2] <= 38)
        self.assertFalse(a["reachability_checked"])

    def test_goal_offset_and_bounds(self):
        cfg = config_dict(parser().parse_args(["--goal-offset", "5", "2", "-3"]))
        g = sample_goal([10, 20, -30], cfg, 1)
        self.assertEqual(g["position_ned_m"], [15, 22, -33])
        cfg["goal_offset"] = [0, 0, 100]
        with self.assertRaises(ValueError):
            sample_goal([0, 0, -30], cfg, 1)

    def test_action_clipping_slew_and_world_command(self):
        cfg = config_dict(parser().parse_args(["--noise-scale", "0"]))
        obs = observation(state(yaw=math.pi / 2))
        cmd = BehaviorPolicy(cfg, 1).command(obs, [0.0] * 4)
        self.assertAlmostEqual(cmd["applied"][1], -0.4)
        for i, value in enumerate(cmd["applied"]):
            self.assertLessEqual(abs(value), ACTION_SCALES[i])
            self.assertAlmostEqual(cmd["normalized_applied"][i], value / ACTION_SCALES[i])
        expected = heading_to_world(cmd["applied"][:3], math.pi / 2)
        self.assertEqual(cmd["api"]["velocity_ned_m_s"], expected)
        self.assertGreater(cmd["api"]["duration_s"], cfg["dt"])

    def test_policy_reproducible_and_altitude_guard(self):
        cfg = config_dict(parser().parse_args([]))
        a, b = BehaviorPolicy(cfg, 3), BehaviorPolicy(cfg, 3)
        obs = observation(state([0, 0, -11]))
        self.assertEqual(a.command(obs, [0] * 4), b.command(obs, [0] * 4))
        cmd = a.command(obs, [0] * 4)
        self.assertIn("lower_altitude_recovery", cmd["altitude_guards"])
        self.assertLess(cmd["applied"][2], 0)

    def test_measurements_use_actual_simulation_time(self):
        cfg = config_dict(parser().parse_args([]))
        before = observation()
        after = observation(state([1, 0, -30]))
        cmd = BehaviorPolicy(cfg, 4).command(before, [0] * 4)
        m = transition_measurements(before, after, cmd, 0.25, 0.0)
        self.assertEqual(m["time_alive_increment_sim_s"], 0.25)
        self.assertEqual(m["average_translation_speed_m_s"], 4.0)
        self.assertAlmostEqual(m["goal_progress_rate_m_s"], m["goal_progress_3d_m"] / 0.25)
        self.assertNotIn("reward", m)
        self.assertNotIn("goal_progress_3d_m", before["goal"])

    def test_validation_rejects_bad_arguments(self):
        p = parser()
        for options in (["--dt", "0"], ["--noise-tau", "0"],
                        ["--goal-ned", "0", "0", "nan"], ["--steps", "-1"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validate_args(p.parse_args(options), p)

    def test_scalar_reward_guard(self):
        no_scalar_reward({"measurements": {"time_alive_increment_sim_s": 0.2}})
        with self.assertRaises(ValueError):
            no_scalar_reward({"transition": {"reward": 1.0}})
        with self.assertRaises(ValueError):
            no_scalar_reward({"distance": float("nan")})

    def test_atomic_session_and_error_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            write_json(session / "ready.json", {"ok": True})
            self.assertEqual(read_json(session / "ready.json"), {"ok": True})
            write_json(session / "error.json", {"process": "record", "message": "camera failed"})
            with self.assertRaisesRegex(RuntimeError, "camera failed"):
                wait_json(session / "missing.json", session, timeout=0.1)
            (session / "error.json").unlink()
            (session / "STOP").touch()
            with self.assertRaises(InterruptedError):
                wait_json(session / "missing.json", session, timeout=0.1)


if __name__ == "__main__":
    unittest.main()

