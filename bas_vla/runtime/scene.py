"""LIBERO layout, camera, and explicit joint-state interventions.

Layout translates movable object free joints together. Camera jitter changes
only the named camera pose; angles use degrees and quaternions use wxyz.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import numpy as np


def _joint_values(values):
    if not isinstance(values, dict) or not values:
        raise ValueError("joint_qpos must be a nonempty mapping of joint names to explicit values")
    for name, value in values.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("joint names must be nonempty strings")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"joint {name!r} requires a finite numeric scalar")
    return {name: float(value) for name, value in values.items()}


def load_joint_state_config(path):
    """Load {"joint_qpos": {actual_joint_name: explicit_qpos, ...}}."""
    if path is None:
        return None
    def unique_pairs(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate joint-state config key: {name}")
            result[name] = value
        return result
    payload = json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs)
    if not isinstance(payload, dict) or set(payload) != {"joint_qpos"}:
        raise ValueError("joint-state config must contain exactly the joint_qpos mapping")
    return _joint_values(payload["joint_qpos"])


def apply_joint_state(env, initial_state, joint_qpos):
    """Set named scalar hinge/slide qpos, honoring enabled model limits.

    Hinge values are radians; slide values use model length units. Names and
    values come from the task assets. Time, other qpos, and qvel are preserved.
    """
    values = _joint_values(joint_qpos)
    state = np.array(initial_state, dtype=np.float64, copy=True)
    model = env.sim.model
    if state.ndim != 1 or state.size != 1 + model.nq + model.nv or not np.isfinite(state).all():
        raise ValueError("expected a finite flattened LIBERO [time, qpos, qvel] state")
    updates = []
    for name, value in values.items():
        try:
            joint = model.joint_name2id(name)
        except (KeyError, ValueError) as error:
            raise ValueError(f"unknown joint: {name}") from error
        if not isinstance(joint, Integral) or not 0 <= joint < len(model.jnt_type):
            raise ValueError(f"unknown joint: {name}")
        # MuJoCo joint enums: free=0, ball=1, slide=2, hinge=3.
        if model.jnt_type[joint] not in (2, 3):
            raise ValueError(f"joint {name!r} must be a scalar hinge or slide")
        address = model.jnt_qposadr[joint]
        end = model.jnt_qposadr[joint + 1] if joint + 1 < len(model.jnt_type) else model.nq
        if not isinstance(address, Integral) or not 0 <= address < model.nq or end - address != 1:
            raise ValueError(f"joint {name!r} has a non-scalar qpos address")
        if model.jnt_limited[joint]:
            lower, upper = model.jnt_range[joint]
            if not np.isfinite([lower, upper]).all() or lower > upper:
                raise ValueError(f"joint {name!r} has invalid model limits")
            if not lower <= value <= upper:
                raise ValueError(f"joint {name!r} value {value} exceeds model limits [{lower}, {upper}]")
        updates.append((1 + address, value))
    for address, value in updates:
        state[address] = value
    return state


@dataclass(frozen=True)
class SceneVariation:
    mode: str = "clean"
    layout_dx: float = .03
    layout_dy: float = .02
    cam_dx: float = .02
    cam_dy: float = .05
    cam_dz: float = 0.
    cam_yaw_deg: float = 5.
    cam_pitch_deg: float = -3.
    camera_name: str = "agentview"
    joint_qpos: dict[str, float] | None = None

    def __post_init__(self):
        if self.mode not in {"clean", "layout_shift", "view_jitter"}:
            raise ValueError(f"unknown scene variation: {self.mode}")
        if not all(np.isfinite(value) for key, value in asdict(self).items() if key not in {"mode", "camera_name", "joint_qpos"}):
            raise ValueError("scene variation parameters must be finite")
        if self.joint_qpos is not None:
            object.__setattr__(self, "joint_qpos", _joint_values(self.joint_qpos))


def add_scene_variation_arguments(parser):
    parser.add_argument("--scene-variation", choices=("clean", "layout_shift", "view_jitter"), default="clean")
    for field, default in asdict(SceneVariation()).items():
        if field not in {"mode", "camera_name", "joint_qpos"}:
            parser.add_argument("--" + field.replace("_", "-"), type=float, default=default)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--joint-state-config", type=Path,
        help="JSON with explicit joint_qpos values for named hinge/slide joints")


def scene_variation_from_args(args):
    fields = asdict(SceneVariation())
    values = {key: getattr(args, "scene_variation" if key == "mode" else key) for key in fields if key != "joint_qpos"}
    values["joint_qpos"] = load_joint_state_config(getattr(args, "joint_state_config", None))
    return SceneVariation(**values)


def get_free_joint_qpos_ranges(env):
    ranges = {}
    for name, obj in env.env.objects_dict.items():
        for joint_name in getattr(obj, "joints", []) or []:
            address = env.sim.model.get_joint_qpos_addr(joint_name)
            if isinstance(address, tuple) and address[1] - address[0] == 7:
                ranges[name] = address
                break
    return ranges


def apply_layout_shift(env, initial_state, dx, dy):
    shifted = np.array(initial_state, dtype=np.float64, copy=True)
    if shifted.ndim != 1 or not np.isfinite(shifted).all() or not np.isfinite([dx, dy]).all():
        raise ValueError("layout shift needs a finite flat simulator state and offsets")
    # LIBERO uses flattened MuJoCo [time, qpos, qvel] states. Fail rather than
    # interpreting actuator/user-state tails as additional leading state data.
    offset = shifted.size - (env.sim.model.nq + env.sim.model.nv)
    if offset != 1:
        raise ValueError("expected a flattened LIBERO [time, qpos, qvel] state")
    ranges = get_free_joint_qpos_ranges(env)
    if not ranges:
        raise ValueError("layout shift found no movable object free joints")
    for start, _ in set(ranges.values()):
        shifted[offset + start] += dx
        shifted[offset + start + 1] += dy
    return shifted


def get_camera_id(env, camera_name="agentview"):
    try:
        return env.sim.model.camera_name2id(camera_name)
    except AttributeError:
        return [env.sim.model.camera_id2name(i) for i in range(env.sim.model.ncam)].index(camera_name)


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def quat_from_euler_xyz(roll_deg, pitch_deg, yaw_deg):
    roll, pitch, yaw = [math.radians(value) / 2 for value in (roll_deg, pitch_deg, yaw_deg)]
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


class SceneIntervention:
    """Apply after reset, before set_init_state; reuse per environment.

    Capturing the original camera pose once prevents repeated episodes from
    accumulating pose offsets when a simulator retains model edits on reset.
    """
    def __init__(self, env, spec: SceneVariation):
        self.spec = spec
        if spec.mode == "view_jitter":
            camera = get_camera_id(env, spec.camera_name)
            self.base_pos = np.array(env.sim.model.cam_pos[camera], copy=True)
            self.base_quat = np.array(env.sim.model.cam_quat[camera], copy=True)

    def apply(self, env, initial_state):
        spec = self.spec
        state = np.array(initial_state, copy=True)
        if spec.joint_qpos is not None:
            state = apply_joint_state(env, state, spec.joint_qpos)
        if spec.mode == "layout_shift":
            state = apply_layout_shift(env, state, spec.layout_dx, spec.layout_dy)
        elif spec.mode == "view_jitter":
            camera = get_camera_id(env, spec.camera_name)
            env.sim.model.cam_pos[camera] = self.base_pos + [spec.cam_dx, spec.cam_dy, spec.cam_dz]
            env.sim.model.cam_quat[camera] = quat_mul_wxyz(
                quat_from_euler_xyz(0., spec.cam_pitch_deg, spec.cam_yaw_deg), self.base_quat)
            env.sim.forward()
        return state

    def metadata(self):
        result = asdict(self.spec)
        if self.spec.mode == "view_jitter":
            result.update(base_camera_position=self.base_pos.tolist(), base_camera_quaternion=self.base_quat.tolist())
        return result
