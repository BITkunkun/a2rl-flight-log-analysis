"""Strict signal mapping and identical segment selection for all entry points."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/analysis_config.json"


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path, track_override=None):
    config = json.loads(resolve_path(path).read_text(encoding="utf-8-sig"))
    if track_override:
        config["track_config"] = track_override
    track = json.loads(resolve_path(config["track_config"]).read_text(encoding="utf-8-sig"))
    gates = np.asarray(track["gates"], dtype=float)
    if gates.ndim != 2 or gates.shape[1] != 4 or len(gates) == 0 or not np.isfinite(gates).all():
        raise ValueError("赛道 gates 必须为非空 [x,y,z_NED,yaw_度] 数组")
    gates[:, 3] = np.deg2rad(gates[:, 3])
    return config, gates, [f"Gate {i:02d}" for i in range(1, len(gates) + 1)]


def vector(prefix, width=3):
    return [f"{prefix}[{i}]" for i in range(width)]


def quaternion_euler(q):
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm < 1e-9):
        raise ValueError("姿态存在无效的零四元数")
    w, x, y, z = (q / norm[:, None]).T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.column_stack([roll, pitch, yaw])


def signal_sources(header, kind):
    if kind == "HIL" and "x_ned" in header:
        return {
            "position": [f"{a}_ned" for a in "xyz"], "velocity": [f"v{a}_ned" for a in "xyz"],
            "attitude": ["qw", "qx", "qy", "qz"], "attitude_format": "quaternion_wxyz",
            "command": [f"motor_{i}" for i in range(1, 5)],
            "rotation": [f"rpm_{i}" for i in range(1, 5)],
        }
    if kind == "HIL" and "pos[0]" in header and "extPos[0]" not in header:
        return {
            "position": vector("pos"), "velocity": vector("vel"),
            "attitude": vector("quat", 4), "attitude_format": "quaternion_wxyz",
            "command": vector("command", 4), "rotation": vector("omega", 4),
        }
    return {
        "position": vector("extPos"), "velocity": vector("extVel"),
        "attitude": vector("extAtt"), "attitude_format": "euler_rad",
        "command": vector("motor", 4), "rotation": vector("omega", 4),
    }


def load_dataset(kind, settings, control_flags):
    path = resolve_path(settings["csv"])
    header = pd.read_csv(path, nrows=0).columns
    sources = signal_sources(header, kind)
    time_options = ("sim_s", "timeS", "timeUs") if kind == "HIL" else ("timeS", "timeUs")
    time_col = next((col for col in time_options if col in header), None)
    if time_col is None:
        raise ValueError(f"{path}: 缺少时间字段 {time_options}")
    required = [time_col, *sources["position"], *sources["velocity"], *sources["attitude"]]
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"{path}: 缺少必要字段 {', '.join(missing)}")
    optional = [*sources["command"], *sources["rotation"],
                "nn_mode", "flightModeFlags", "armed", "position_mode"]
    raw = pd.read_csv(path, usecols=required + [col for col in optional if col in header])
    raw = raw.apply(pd.to_numeric, errors="coerce")
    if len(raw) < 2:
        raise ValueError(f"{path}: 日志记录不足两行")
    clock = raw[time_col].to_numpy(dtype=float) / (1_000_000 if time_col == "timeUs" else 1)
    if not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
        raise ValueError(f"{path}: {time_col} 必须有效且严格递增")
    log_time = clock - clock[0]

    if kind == "HIL" and "nn_mode" in raw:
        if not raw.nn_mode.isin([0, 1]).all():
            raise ValueError(f"{path}: nn_mode 必须为 0 或 1")
        mask, selection = raw.nn_mode.eq(1).to_numpy(), "nn_mode == 1"
    elif kind == "REAL" or (kind == "HIL" and "flightModeFlags" in raw):
        if "flightModeFlags" not in raw or not control_flags:
            raise ValueError(f"{path}: 缺少 flightModeFlags 或配置的 control_flags")
        if not np.isfinite(raw.flightModeFlags).all():
            raise ValueError(f"{path}: flightModeFlags 存在无效值")
        mask = raw.flightModeFlags.isin(control_flags).to_numpy()
        selection = f"flightModeFlags in {control_flags}"
    elif kind == "HIL":
        start = settings.get("nn_start_s")
        if start is None or not np.isfinite(start) or start < 0:
            raise ValueError(f"{path}: HIL 缺少 NN 模式标记，请在配置中明确 nn_start_s（相对日志起点）")
        mask, selection = log_time >= start, f"explicit NN start at log + {start} s"
    else:
        mask, selection = np.ones(len(raw), dtype=bool), "full SITL log"
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        raise ValueError(f"{path}: 分析段不足两行，选择规则为 {selection}")
    if np.any(np.diff(indices) != 1):
        raise ValueError(f"{path}: 存在多个不连续控制段，请提供本次实验的连续日志")
    selected = raw.iloc[indices].reset_index(drop=True)
    if not np.isfinite(selected[required].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: 分析段必要字段含无效值")
    time = clock[indices]
    frame = pd.DataFrame({"time_s": time - time[0], "log_time_s": log_time[indices]})
    metadata = {}
    for group, target, unit in (("position", "pos_{}_m", "m"), ("velocity", "vel_{}_mps", "m/s")):
        for axis, source in zip("xyz", sources[group]):
            column = target.format(axis)
            frame[column] = selected[source]
            metadata[column] = (source, unit)
    angles = selected[sources["attitude"]].to_numpy(dtype=float)
    if sources["attitude_format"] == "quaternion_wxyz":
        angles = quaternion_euler(angles)
    for i, axis in enumerate(("roll", "pitch", "yaw")):
        frame[f"{axis}_deg"] = np.rad2deg(angles[:, i])
        source = sources["attitude"][i] if sources["attitude_format"] == "euler_rad" else ", ".join(sources["attitude"])
        metadata[f"{axis}_deg"] = (source, "deg")
    frame["yaw_deg"] = (frame.yaw_deg + 180) % 360 - 180
    frame["yaw_unwrapped_deg"] = np.rad2deg(np.unwrap(np.deg2rad(frame.yaw_deg)))
    frame["height_m"] = -frame.pos_z_m
    frame["speed_xy_mps"] = np.hypot(frame.vel_x_mps, frame.vel_y_mps)
    frame["speed_3d_mps"] = np.hypot(frame.speed_xy_mps, frame.vel_z_mps)
    metadata.update({
        "yaw_unwrapped_deg": ("unwrap(yaw)", "deg"), "height_m": ("-pos_z_m", "m"),
        "speed_xy_mps": ("hypot(vel_x_mps, vel_y_mps)", "m/s"),
        "speed_3d_mps": ("norm(velocity)", "m/s"),
    })
    for group, target in (("command", "command"), ("rotation", "omega")):
        for i, source in enumerate(sources[group]):
            if source in selected:
                column = f"{target}_{i}"
                frame[column] = selected[source]
                metadata[column] = (source, "raw")
    for source in ("nn_mode", "flightModeFlags", "armed", "position_mode"):
        if source in selected:
            frame[source] = selected[source]
    delta = np.diff(time)
    profile = {
        "dataset": kind, "source": str(path.resolve()), "rows": len(frame),
        "full_log_rows": len(raw), "full_log_duration_s": float(log_time[-1]),
        "selection": selection, "selection_start_s": float(log_time[indices[0]]),
        "selection_end_s": float(log_time[indices[-1]]),
        "clock_start_s": float(time[0]), "clock_end_s": float(time[-1]),
        "time_source": time_col, "duration_s": float(frame.time_s.iloc[-1]),
        "median_sample_interval_s": float(np.median(delta)), "max_sample_interval_s": float(delta.max()),
        "position_columns": sources["position"], "velocity_columns": sources["velocity"],
        "attitude_columns": sources["attitude"], "attitude_format": sources["attitude_format"],
        "command_columns": [col for col in sources["command"] if col in selected],
        "rotation_columns": [col for col in sources["rotation"] if col in selected],
        "coordinates": "NED; height = -z", "signal_metadata": metadata,
    }
    return frame, profile


def window_frame(frame, duration):
    result = frame.loc[frame.time_s <= duration].copy()
    if len(result) < 2:
        raise ValueError("共同窗口内少于两行数据")
    return result
