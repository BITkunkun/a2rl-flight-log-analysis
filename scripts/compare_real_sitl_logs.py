from __future__ import annotations
# python scripts\compare_real_sitl_logs.py --config configs\real_sitl_compare_config.json
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from anomaly_annotations import finish_annotated_layout, mark_time_interval, mark_trajectory_interval, save_figure

from track_3d import (
    DEFAULT_GATE_LABELS,
    DEFAULT_GATES_NED,
    evaluate_gate_passages,
    normalize_gates,
    plot_trajectory_gate_3d,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL = ROOT / "data" / "raw" / "real" / "circle_0704.csv"
DEFAULT_SITL = ROOT / "data" / "raw" / "sitl" / "LOG_SITL_0628lyh.csv"
DEFAULT_OUT_ROOT = ROOT / "reports"
DEFAULT_OUT_NAME = "real_sitl_comparison"
DEFAULT_CONFIG = ROOT / "configs" / "real_sitl_compare_config.json"
REAL_CONTROL_FLAGS = (12803, 12291)
SITL_MOTION_SPEED_MPS = 0.30
SITL_MOTION_DISPLACEMENT_M = 0.20
MAX_PLOT_POINTS = 7000
CMD_SATURATION = 0.95
GATES = DEFAULT_GATES_NED.copy()
GATE_LABELS = DEFAULT_GATE_LABELS.copy()

@dataclass
class Dataset:
    key: str
    name: str
    domain: str
    color: str
    path: Path
    full: pd.DataFrame
    selected: pd.DataFrame
    profile: dict
    segment_note: str
    anomaly_interval: dict | None = None


@dataclass
class OutputPaths:
    out_dir: Path
    figures_dir: Path
    tables_dir: Path
    processed_dir: Path
    doc_path: Path
    manifest_path: Path


def configure_matplotlib() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["savefig.facecolor"] = "white"


def parse_flags(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return REAL_CONTROL_FLAGS
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def parse_flags_value(value: Any) -> tuple[int, ...]:
    if value is None:
        return REAL_CONTROL_FLAGS
    if isinstance(value, str):
        return parse_flags(value)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return (int(value),)


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cfg_value(config: dict[str, Any], key: str, cli_value: Any, default: Any) -> Any:
    return cli_value if cli_value is not None else config.get(key, default)


def normalize_gate_labels(labels: list[str], gate_count: int) -> list[str]:
    if gate_count <= 0:
        return []
    normalized = [str(item).strip() for item in labels if str(item).strip()]
    if len(normalized) == gate_count:
        return normalized
    original_count = len(normalized)
    if len(normalized) < gate_count:
        normalized.extend(f"Gate {idx + 1}" for idx in range(len(normalized), gate_count))
    else:
        normalized = normalized[:gate_count]
    print(
        "Warning: gate_labels length does not match gates length; "
        f"adjusted labels from {original_count} to {gate_count}."
    )
    return normalized


def resolve_workspace_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def vector_cols(columns: set[str], prefix: str, n: int) -> list[str]:
    cols = [f"{prefix}[{idx}]" for idx in range(n)]
    return cols if all(col in columns for col in cols) else []


def first_vector(columns: set[str], prefixes: Iterable[str], n: int) -> tuple[str | None, list[str]]:
    for prefix in prefixes:
        cols = vector_cols(columns, prefix, n)
        if cols:
            return prefix, cols
    return None, []


def wanted_columns(columns: list[str]) -> list[str]:
    available = set(columns)
    wanted = {
        "timeUs",
        "timeS",
        "loopIteration",
        "flightModeFlags",
        "stateFlags",
        "failsafePhase",
        "rxSignalReceived",
        "rxFlightChannelsValid",
        "baroAlt",
    }
    for prefix in (
        "pos",
        "extPos",
        "ekf_pos",
        "posSp",
        "vel",
        "extVel",
        "ekf_vel",
        "velSp",
        "quat",
        "quatSp",
        "extAtt",
        "ekf_att",
        "gyroADC",
        "gyroSp",
        "motor",
        "omegaUnfiltered",
        "rcCommand",
        "setpoint",
    ):
        n = 4 if prefix in {"quat", "quatSp", "motor", "omegaUnfiltered", "rcCommand", "setpoint"} else 3
        wanted.update(f"{prefix}[{idx}]" for idx in range(n))
    return [col for col in columns if col in wanted and col in available]


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_euler_rad(q: np.ndarray) -> np.ndarray:
    q = q.astype(float)
    norm = np.linalg.norm(q, axis=1)
    good = np.isfinite(norm) & (norm > 1e-9)
    qn = np.zeros_like(q)
    qn[good] = q[good] / norm[good, None]
    w, x, y, z = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    out = np.vstack([roll, pitch, yaw]).T
    out[~good] = np.nan
    return out


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def stats(values: pd.Series | np.ndarray) -> dict[str, float]:
    arr = finite(values)
    if len(arr) == 0:
        return {key: float("nan") for key in ("mean", "std", "min", "p05", "p50", "p95", "max")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def rmse(values: pd.Series | np.ndarray) -> float:
    arr = finite(values)
    return float(np.sqrt(np.mean(arr**2))) if len(arr) else float("nan")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def lag_ms(cmd: np.ndarray, rpm: np.ndarray, dt_s: float, max_lag_ms: float = 250.0) -> float:
    mask = np.isfinite(cmd) & np.isfinite(rpm)
    cmd = cmd[mask]
    rpm = rpm[mask]
    if len(cmd) < 20 or dt_s <= 0:
        return float("nan")
    cmd = cmd - np.mean(cmd)
    rpm = rpm - np.mean(rpm)
    if np.std(cmd) == 0 or np.std(rpm) == 0:
        return float("nan")
    max_lag = max(1, int(max_lag_ms / (dt_s * 1000.0)))
    best_corr = -np.inf
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = cmd[-lag:], rpm[:lag]
        elif lag > 0:
            x, y = cmd[:-lag], rpm[lag:]
        else:
            x, y = cmd, rpm
        if len(x) < 10:
            continue
        score = corr(x, y)
        if np.isfinite(score) and score > best_corr:
            best_corr = score
            best_lag = lag
    return float(best_lag * dt_s * 1000.0)


def percentile(values: pd.Series | np.ndarray, q: float) -> float:
    arr = finite(values)
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    idx = np.unique(np.linspace(0, len(df) - 1, max_points).astype(int))
    return df.iloc[idx].copy()


def normalized_time(df: pd.DataFrame) -> pd.Series:
    duration = float(df["segment_time_s"].iloc[-1] - df["segment_time_s"].iloc[0])
    if duration <= 0:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return (df["segment_time_s"] - float(df["segment_time_s"].iloc[0])) / duration


def comparison_time_limit(datasets: list[Dataset], requested_s: float | None) -> float:
    if requested_s is not None and requested_s > 0:
        return float(requested_s)
    shortest = min(float(ds.selected["segment_time_s"].iloc[-1]) for ds in datasets)
    if shortest <= 0:
        return 1.0
    # Keep the entire shorter segment visible, but round up to a clean ruler.
    if shortest <= 15:
        step = 2.0
    elif shortest <= 40:
        step = 5.0
    else:
        step = 10.0
    return float(math.ceil(shortest / step) * step)


def trim_time_window(df: pd.DataFrame, x_max_s: float) -> pd.DataFrame:
    trimmed = df[df["segment_time_s"] <= x_max_s].copy()
    return trimmed if not trimmed.empty else df.iloc[:1].copy()


def normalize_log_frame(path: Path, domain: str = "real") -> tuple[pd.DataFrame, dict]:
    columns = read_header(path)
    raw = pd.read_csv(path, usecols=wanted_columns(columns))
    available = set(raw.columns)
    if "timeUs" in raw:
        time_s = (pd.to_numeric(raw["timeUs"], errors="coerce") - float(raw["timeUs"].iloc[0])) / 1_000_000.0
    elif "timeS" in raw:
        time_s = pd.to_numeric(raw["timeS"], errors="coerce") - float(raw["timeS"].iloc[0])
    else:
        raise ValueError(f"{path} needs timeUs or timeS.")

    pos_sources = ("extPos", "pos", "ekf_pos") if domain == "sitl" else ("pos", "extPos", "ekf_pos")
    vel_sources = ("extVel", "vel", "ekf_vel") if domain == "sitl" else ("vel", "extVel", "ekf_vel")
    pos_prefix, pos_cols = first_vector(available, pos_sources, 3)
    vel_prefix, vel_cols = first_vector(available, vel_sources, 3)
    pos_sp_prefix, pos_sp_cols = first_vector(available, ("posSp",), 3)
    vel_sp_prefix, vel_sp_cols = first_vector(available, ("velSp",), 3)
    _, quat_cols = first_vector(available, ("quat",), 4)
    _, quat_sp_cols = first_vector(available, ("quatSp",), 4)
    att_prefix, att_cols = first_vector(available, ("extAtt", "ekf_att"), 3)

    if not pos_cols:
        raise ValueError(f"{path} has no position vector columns.")
    if not vel_cols:
        raise ValueError(f"{path} has no velocity vector columns.")
    if not quat_cols and not att_cols:
        raise ValueError(f"{path} has no attitude columns.")

    df = pd.DataFrame({"time_s": time_s})
    for axis, col in zip(("x", "y", "z"), pos_cols):
        df[f"pos_{axis}_m"] = pd.to_numeric(raw[col], errors="coerce")
    for axis, col in zip(("x", "y", "z"), vel_cols):
        df[f"vel_{axis}_mps"] = pd.to_numeric(raw[col], errors="coerce")
    for axis, col in zip(("x", "y", "z"), pos_sp_cols):
        df[f"pos_sp_{axis}_m"] = pd.to_numeric(raw[col], errors="coerce")
    for axis, col in zip(("x", "y", "z"), vel_sp_cols):
        df[f"vel_sp_{axis}_mps"] = pd.to_numeric(raw[col], errors="coerce")

    if quat_cols:
        euler = quat_to_euler_rad(raw[quat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
        attitude_source = quat_cols
    else:
        att = raw[att_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.nanmedian(np.abs(att)) > math.pi * 1.5:
            att = np.deg2rad(att)
        euler = att
        attitude_source = att_cols
    df["roll_rad"] = euler[:, 0]
    df["pitch_rad"] = euler[:, 1]
    df["yaw_rad"] = wrap_pi(euler[:, 2])
    df["roll_deg"] = np.rad2deg(df["roll_rad"])
    df["pitch_deg"] = np.rad2deg(df["pitch_rad"])
    df["yaw_deg"] = np.rad2deg(df["yaw_rad"])
    df["yaw_unwrapped_deg"] = np.rad2deg(np.unwrap(df["yaw_rad"].to_numpy(dtype=float)))

    if quat_sp_cols:
        quat_sp = raw[quat_sp_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        norm = np.linalg.norm(quat_sp, axis=1)
        if np.nanmedian(norm[np.isfinite(norm)]) > 0.5:
            euler_sp = quat_to_euler_rad(quat_sp)
            df["yaw_sp_rad"] = wrap_pi(euler_sp[:, 2])
            df["yaw_sp_deg"] = np.rad2deg(df["yaw_sp_rad"])
            df["yaw_err_deg"] = np.rad2deg(wrap_pi(df["yaw_rad"].to_numpy() - df["yaw_sp_rad"].to_numpy()))

    z_reference = df["pos_z_m"]
    if "pos_sp_z_m" in df:
        z_reference = pd.concat([df["pos_z_m"], df["pos_sp_z_m"]], axis=0)
    z_sign = -1.0 if float(np.nanmedian(z_reference)) < 0.0 else 1.0
    df["height_m"] = z_sign * df["pos_z_m"]
    if "pos_sp_z_m" in df:
        df["height_sp_m"] = z_sign * df["pos_sp_z_m"]
        df["pos_err_x_m"] = df["pos_x_m"] - df["pos_sp_x_m"]
        df["pos_err_y_m"] = df["pos_y_m"] - df["pos_sp_y_m"]
        df["height_err_m"] = df["height_m"] - df["height_sp_m"]
        df["pos_err_xy_m"] = np.hypot(df["pos_err_x_m"], df["pos_err_y_m"])
        df["pos_err_3d_m"] = np.sqrt(df["pos_err_xy_m"] ** 2 + df["height_err_m"] ** 2)

    df["speed_xy_mps"] = np.hypot(df["vel_x_mps"], df["vel_y_mps"])
    df["speed_3d_mps"] = np.sqrt(df["vel_x_mps"] ** 2 + df["vel_y_mps"] ** 2 + df["vel_z_mps"] ** 2)
    df["velocity_heading_rad"] = np.arctan2(df["vel_y_mps"], df["vel_x_mps"])
    df["yaw_velocity_error_deg"] = np.rad2deg(wrap_pi(df["yaw_rad"].to_numpy() - df["velocity_heading_rad"].to_numpy()))
    if "vel_sp_x_mps" in df and "vel_sp_y_mps" in df:
        df["speed_sp_xy_mps"] = np.hypot(df["vel_sp_x_mps"], df["vel_sp_y_mps"])
        df["speed_xy_err_mps"] = df["speed_xy_mps"] - df["speed_sp_xy_mps"]

    for idx in range(4):
        motor_col = f"motor[{idx}]"
        omega_col = f"omegaUnfiltered[{idx}]"
        if motor_col in raw:
            df[f"motor_{idx}"] = pd.to_numeric(raw[motor_col], errors="coerce")
        if omega_col in raw:
            df[f"omega_{idx}_rpm"] = pd.to_numeric(raw[omega_col], errors="coerce")

    for idx, axis in enumerate(("x", "y", "z")):
        gyro_col = f"gyroADC[{idx}]"
        if gyro_col in raw:
            gyro = pd.to_numeric(raw[gyro_col], errors="coerce")
            df[f"gyro_{axis}_dps"] = np.rad2deg(gyro) if float(np.nanmedian(np.abs(gyro))) < 20.0 else gyro
        gyro_sp_col = f"gyroSp[{idx}]"
        if gyro_sp_col in raw:
            gyro_sp = pd.to_numeric(raw[gyro_sp_col], errors="coerce")
            df[f"gyro_sp_{axis}_dps"] = np.rad2deg(gyro_sp) if float(np.nanmedian(np.abs(gyro_sp))) < 20.0 else gyro_sp

    for col in ("flightModeFlags", "stateFlags", "failsafePhase", "rxSignalReceived", "rxFlightChannelsValid", "baroAlt"):
        if col in raw:
            df[col] = pd.to_numeric(raw[col], errors="coerce")
    for idx in range(4):
        col = f"rcCommand[{idx}]"
        if col in raw:
            df[f"rc_command_{idx}"] = pd.to_numeric(raw[col], errors="coerce")
        sp_col = f"setpoint[{idx}]"
        if sp_col in raw:
            df[f"setpoint_{idx}"] = pd.to_numeric(raw[sp_col], errors="coerce")

    motor_cols = [f"motor_{idx}" for idx in range(4) if f"motor_{idx}" in df]
    omega_cols = [f"omega_{idx}_rpm" for idx in range(4) if f"omega_{idx}_rpm" in df]
    if motor_cols:
        df["motor_mean"] = df[motor_cols].mean(axis=1)
        df["motor_max"] = df[motor_cols].max(axis=1)
        df["motor_min"] = df[motor_cols].min(axis=1)
        df["motor_spread"] = df["motor_max"] - df["motor_min"]
    if omega_cols:
        df["omega_mean_rpm"] = df[omega_cols].mean(axis=1)
        df["omega_max_rpm"] = df[omega_cols].max(axis=1)
        df["omega_min_rpm"] = df[omega_cols].min(axis=1)
        df["omega_spread_rpm"] = df["omega_max_rpm"] - df["omega_min_rpm"]

    profile = {
        "source": str(path),
        "rows_raw": int(len(raw)),
        "columns_raw": int(len(columns)),
        "duration_s": float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]),
        "approx_hz": float((len(df) - 1) / max(df["time_s"].iloc[-1] - df["time_s"].iloc[0], 1e-9)),
        "position_columns": pos_cols,
        "velocity_columns": vel_cols,
        "position_setpoint_columns": pos_sp_cols,
        "velocity_setpoint_columns": vel_sp_cols,
        "attitude_source": attitude_source,
        "yaw_setpoint_source": quat_sp_cols,
        "position_prefix": pos_prefix,
        "velocity_prefix": vel_prefix,
        "position_setpoint_prefix": pos_sp_prefix,
        "velocity_setpoint_prefix": vel_sp_prefix,
        "attitude_prefix": att_prefix,
        "height_convention": "NED z down, height = -z" if z_sign < 0 else "z up, height = z",
    }
    if "flightModeFlags" in df:
        counts = df["flightModeFlags"].fillna(-1).astype(int).value_counts().sort_index()
        profile["flight_mode_counts"] = {int(k): int(v) for k, v in counts.items()}
    return df, profile


def select_real_segment(df: pd.DataFrame, flags: tuple[int, ...]) -> tuple[np.ndarray, str]:
    if "flightModeFlags" in df:
        mask = df["flightModeFlags"].fillna(-1).astype(int).isin(flags).to_numpy()
        if mask.any():
            return mask, f"实飞按 flightModeFlags in {list(flags)} 截取神经网络控制段"
    moving = df["speed_xy_mps"].to_numpy(dtype=float) > SITL_MOTION_SPEED_MPS
    return moving, f"实飞未找到指定 flags，退化为速度 > {SITL_MOTION_SPEED_MPS:.2f} m/s 的运动段"


def select_sitl_segment(df: pd.DataFrame, speed_threshold: float, displacement_threshold: float) -> tuple[np.ndarray, str]:
    speed = df["speed_xy_mps"].to_numpy(dtype=float)
    pos = df[["pos_x_m", "pos_y_m"]].to_numpy(dtype=float)
    displacement = np.linalg.norm(pos - pos[0], axis=1)
    candidates = (speed > speed_threshold) | (displacement > displacement_threshold)
    idx = np.flatnonzero(candidates)
    if len(idx) == 0:
        return np.ones(len(df), dtype=bool), "SITL 未检测到明显运动，全段用于分析"
    mask = np.zeros(len(df), dtype=bool)
    mask[idx[0] : idx[-1] + 1] = True
    return (
        mask,
        f"SITL 的 flightModeFlags 全为 0，按速度 > {speed_threshold:.2f} m/s 或偏离起点 > {displacement_threshold:.2f} m 自动截取运动段",
    )


def apply_segment(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    selected = df.loc[mask].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError("Selected segment is empty.")
    selected["segment_time_s"] = selected["time_s"] - float(selected["time_s"].iloc[0])
    selected["norm_time"] = normalized_time(selected)
    return selected


def load_dataset(
    key: str,
    name: str,
    domain: str,
    color: str,
    path: Path,
    real_flags: tuple[int, ...],
    sitl_speed_threshold: float,
    sitl_displacement_threshold: float,
) -> Dataset:
    full, profile = normalize_log_frame(path, domain=domain)
    if domain == "real":
        mask, note = select_real_segment(full, real_flags)
    else:
        mask, note = select_sitl_segment(full, sitl_speed_threshold, sitl_displacement_threshold)
    selected = apply_segment(full, mask)
    profile["rows_selected"] = int(len(selected))
    profile["selected_start_s"] = float(selected["time_s"].iloc[0])
    profile["selected_end_s"] = float(selected["time_s"].iloc[-1])
    profile["selected_duration_s"] = float(selected["segment_time_s"].iloc[-1])
    profile["selected_row_fraction"] = float(len(selected) / max(len(full), 1))
    if "flightModeFlags" in selected:
        counts = selected["flightModeFlags"].fillna(-1).astype(int).value_counts().sort_index()
        profile["selected_flight_mode_counts"] = {int(k): int(v) for k, v in counts.items()}
    return Dataset(key, name, domain, color, path, full, selected, profile, note)


def circular_offset(target: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    good = np.asarray(mask, dtype=bool) & np.isfinite(target) & np.isfinite(actual)
    if good.sum() < 10:
        return float("nan"), float("nan"), float("nan")
    offset = float(np.angle(np.mean(np.exp(1j * (target[good] - actual[good])))))
    residual = np.abs(np.rad2deg(wrap_pi(target - actual - offset)))[good]
    return float(np.rad2deg(offset)), float(np.percentile(residual, 50)), float(np.percentile(residual, 95))


def count_gate_passes(df: pd.DataFrame, gate_xy: np.ndarray, radius_m: float = 0.55, min_interval_s: float = 0.8) -> int:
    if len(df) < 3:
        return 0
    d = np.hypot(df["pos_x_m"].to_numpy() - gate_xy[0], df["pos_y_m"].to_numpy() - gate_xy[1])
    t = df["segment_time_s"].to_numpy(dtype=float)
    candidates = np.where((d[1:-1] <= d[:-2]) & (d[1:-1] < d[2:]) & (d[1:-1] < radius_m))[0] + 1
    passes: list[tuple[float, float]] = []
    for idx in candidates:
        if not passes or t[idx] - passes[-1][0] > min_interval_s:
            passes.append((float(t[idx]), float(d[idx])))
        elif d[idx] < passes[-1][1]:
            passes[-1] = (float(t[idx]), float(d[idx]))
    return len(passes)


def make_gate_metrics(dataset: Dataset) -> pd.DataFrame:
    df = dataset.selected
    rows = []
    for label, gate in zip(GATE_LABELS, GATES):
        dist = np.hypot(df["pos_x_m"].to_numpy() - gate[0], df["pos_y_m"].to_numpy() - gate[1])
        idx = int(np.nanargmin(dist))
        row = df.iloc[idx]
        rows.append(
            {
                "dataset": dataset.name,
                "domain": dataset.domain,
                "gate": label,
                "gate_x_m": float(gate[0]),
                "gate_y_m": float(gate[1]),
                "gate_z_ned_m": float(gate[2]),
                "min_distance_m": float(dist[idx]),
                "closest_segment_time_s": float(row["segment_time_s"]),
                "closest_x_m": float(row["pos_x_m"]),
                "closest_y_m": float(row["pos_y_m"]),
                "closest_height_m": float(row["height_m"]),
                "speed_xy_mps": float(row["speed_xy_mps"]),
                "yaw_deg": float(row["yaw_deg"]),
                "motor_max": float(row["motor_max"]) if "motor_max" in row else float("nan"),
                "pass_count_lt_0p55m": count_gate_passes(df, gate),
            }
        )
    return pd.DataFrame(rows)


def make_motor_metrics(dataset: Dataset) -> pd.DataFrame:
    df = dataset.selected
    dt_s = float(np.nanmedian(np.diff(df["segment_time_s"].to_numpy(dtype=float)))) if len(df) > 2 else 0.0
    rows = []
    for idx in range(4):
        cmd_col = f"motor_{idx}"
        rpm_col = f"omega_{idx}_rpm"
        if cmd_col not in df or rpm_col not in df:
            continue
        cmd = df[cmd_col].to_numpy(dtype=float)
        rpm = df[rpm_col].to_numpy(dtype=float)
        cmd_stats = stats(cmd)
        rpm_stats = stats(rpm)
        rows.append(
            {
                "dataset": dataset.name,
                "domain": dataset.domain,
                "motor": idx,
                "cmd_mean": cmd_stats["mean"],
                "cmd_p50": cmd_stats["p50"],
                "cmd_p95": cmd_stats["p95"],
                "cmd_max": cmd_stats["max"],
                "cmd_high_sat_rate": float(np.mean(cmd >= CMD_SATURATION)) if len(cmd) else float("nan"),
                "rpm_mean": rpm_stats["mean"],
                "rpm_p50": rpm_stats["p50"],
                "rpm_p95": rpm_stats["p95"],
                "rpm_max": rpm_stats["max"],
                "cmd_rpm_corr": corr(cmd, rpm),
                "best_lag_ms": lag_ms(cmd, rpm, dt_s),
            }
        )
    return pd.DataFrame(rows)


def make_summary(dataset: Dataset, gate_metrics: pd.DataFrame) -> dict[str, float | int | str]:
    df = dataset.selected
    dpos_xy = np.diff(df[["pos_x_m", "pos_y_m"]].to_numpy(dtype=float), axis=0)
    dpos_3d = np.diff(df[["pos_x_m", "pos_y_m", "height_m"]].to_numpy(dtype=float), axis=0)
    path_xy = float(np.nansum(np.linalg.norm(dpos_xy, axis=1))) if len(dpos_xy) else float("nan")
    path_3d = float(np.nansum(np.linalg.norm(dpos_3d, axis=1))) if len(dpos_3d) else float("nan")
    moving = df["speed_xy_mps"].to_numpy(dtype=float) > 0.30
    vel_heading = df["velocity_heading_rad"].to_numpy(dtype=float)
    yaw = df["yaw_rad"].to_numpy(dtype=float)
    yaw_vel_abs = np.abs(np.rad2deg(wrap_pi(yaw - vel_heading)))[moving]
    best_offset, offset_p50, offset_p95 = circular_offset(vel_heading, yaw, moving)
    speed = stats(df["speed_xy_mps"])
    height = stats(df["height_m"])
    motor_max = stats(df["motor_max"]) if "motor_max" in df else {}
    omega_mean = stats(df["omega_mean_rpm"]) if "omega_mean_rpm" in df else {}
    omega_spread = stats(df["omega_spread_rpm"]) if "omega_spread_rpm" in df else {}
    summary: dict[str, float | int | str] = {
        "dataset": dataset.name,
        "domain": dataset.domain,
        "source": str(dataset.path),
        "rows_raw": int(dataset.profile["rows_raw"]),
        "rows_selected": int(dataset.profile["rows_selected"]),
        "raw_duration_s": float(dataset.profile["duration_s"]),
        "selected_start_s": float(dataset.profile["selected_start_s"]),
        "selected_end_s": float(dataset.profile["selected_end_s"]),
        "selected_duration_s": float(dataset.profile["selected_duration_s"]),
        "approx_hz": float(dataset.profile["approx_hz"]),
        "segment_note": dataset.segment_note,
        "position_source": ",".join(dataset.profile.get("position_columns", [])),
        "velocity_source": ",".join(dataset.profile.get("velocity_columns", [])),
        "attitude_source": ",".join(dataset.profile.get("attitude_source", [])),
        "height_convention": str(dataset.profile.get("height_convention", "")),
        "flight_mode_counts": json.dumps(dataset.profile.get("selected_flight_mode_counts", {}), ensure_ascii=False),
        "path_xy_m": path_xy,
        "path_3d_m": path_3d,
        "x_range_m": float(df["pos_x_m"].max() - df["pos_x_m"].min()),
        "y_range_m": float(df["pos_y_m"].max() - df["pos_y_m"].min()),
        "height_min_m": height["min"],
        "height_mean_m": height["mean"],
        "height_max_m": height["max"],
        "speed_mean_mps": speed["mean"],
        "speed_p50_mps": speed["p50"],
        "speed_p95_mps": speed["p95"],
        "speed_max_mps": speed["max"],
        "roll_abs_p95_deg": percentile(np.abs(df["roll_deg"]), 95),
        "pitch_abs_p95_deg": percentile(np.abs(df["pitch_deg"]), 95),
        "yaw_total_rotation_deg": float(df["yaw_unwrapped_deg"].iloc[-1] - df["yaw_unwrapped_deg"].iloc[0]),
        "yaw_velocity_abs_p50_deg": float(np.percentile(yaw_vel_abs, 50)) if len(yaw_vel_abs) else float("nan"),
        "yaw_velocity_abs_p95_deg": float(np.percentile(yaw_vel_abs, 95)) if len(yaw_vel_abs) else float("nan"),
        "best_yaw_to_velocity_offset_deg": best_offset,
        "yaw_velocity_after_offset_p50_deg": offset_p50,
        "yaw_velocity_after_offset_p95_deg": offset_p95,
        "motor_cmd_max_p95": motor_max.get("p95", float("nan")),
        "motor_cmd_max_max": motor_max.get("max", float("nan")),
        "motor_any_high_sat_rate": float(np.mean(df["motor_max"] >= CMD_SATURATION)) if "motor_max" in df else float("nan"),
        "omega_mean_p50_rpm": omega_mean.get("p50", float("nan")),
        "omega_mean_p95_rpm": omega_mean.get("p95", float("nan")),
        "omega_spread_p95_rpm": omega_spread.get("p95", float("nan")),
        "gate_min_distance_median": float(gate_metrics["min_distance_m"].median()),
        "gate_pass_count_min": int(gate_metrics["pass_count_lt_0p55m"].min()),
        "gate_pass_count_total": int(gate_metrics["pass_count_lt_0p55m"].sum()),
    }
    if "pos_err_xy_m" in df:
        summary["position_xy_rmse_m"] = rmse(df["pos_err_xy_m"])
        summary["position_xy_p95_m"] = percentile(df["pos_err_xy_m"], 95)
        summary["height_rmse_m"] = rmse(df["height_err_m"])
    if "speed_xy_err_mps" in df:
        summary["speed_xy_rmse_mps"] = rmse(df["speed_xy_err_mps"])
    if "yaw_err_deg" in df:
        summary["yaw_abs_error_p50_deg"] = percentile(np.abs(df["yaw_err_deg"]), 50)
        summary["yaw_abs_error_p95_deg"] = percentile(np.abs(df["yaw_err_deg"]), 95)
    return summary


def common_xy_limits(datasets: list[Dataset], margin_ratio: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = np.concatenate([finite(ds.selected["pos_x_m"]) for ds in datasets])
    ys = np.concatenate([finite(ds.selected["pos_y_m"]) for ds in datasets])
    xs = np.concatenate([xs, GATES[:, 0]])
    ys = np.concatenate([ys, GATES[:, 1]])
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin, 1.0) * (1.0 + margin_ratio)
    half = span / 2.0
    return (cx - half, cx + half), (cy - half, cy + half)


def common_y_limits(datasets: list[Dataset], col: str, margin_ratio: float = 0.08) -> tuple[float, float]:
    vals = [finite(ds.selected[col]) for ds in datasets if col in ds.selected]
    vals = [v for v in vals if len(v)]
    if not vals:
        return 0.0, 1.0
    merged = np.concatenate(vals)
    ymin, ymax = float(np.min(merged)), float(np.max(merged))
    span = max(ymax - ymin, 1.0)
    return ymin - span * margin_ratio, ymax + span * margin_ratio


def draw_gates(ax) -> None:
    ax.scatter(GATES[:, 0], GATES[:, 1], marker="D", s=70, color="#222222", zorder=5, label="门点")
    labels_by_xy: dict[tuple[float, float], list[str]] = {}
    for label, gate in zip(GATE_LABELS, GATES):
        gx, gy = gate[:2]
        labels_by_xy.setdefault((float(gx), float(gy)), []).append(label)
    for (gx, gy), labels in labels_by_xy.items():
        if len(labels) > 1 and all(label.startswith("Gate ") for label in labels):
            text = f"Gate {'/'.join(label.removeprefix('Gate ') for label in labels)}"
        else:
            text = "/".join(labels)
        ax.text(gx + 0.06, gy + 0.06, text, fontsize=9, weight="bold")


def draw_yaw_arrows(ax, df: pd.DataFrame, color: str, max_arrows: int = 22) -> None:
    if len(df) == 0:
        return
    step = max(1, len(df) // max_arrows)
    q = df.iloc[::step]
    ax.quiver(
        q["pos_x_m"],
        q["pos_y_m"],
        np.cos(q["yaw_rad"]),
        np.sin(q["yaw_rad"]),
        color=color,
        alpha=0.32,
        width=0.003,
        scale=15,
        headwidth=4.0,
        headlength=5.0,
    )


def plot_trajectory(datasets: list[Dataset], out: Path, max_points: int) -> Path:
    xlim, ylim = common_xy_limits(datasets)
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])
    ax_overlay = fig.add_subplot(gs[0, :])
    side_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    for ds in datasets:
        d = downsample(ds.selected, max_points)
        ax_overlay.plot(d["pos_x_m"], d["pos_y_m"], color=ds.color, lw=1.4, label=ds.name)
        draw_yaw_arrows(ax_overlay, d, ds.color, max_arrows=16)
        if ds.anomaly_interval:
            mark_trajectory_interval(ax_overlay, ds.selected, ds.anomaly_interval, ("pos_x_m", "pos_y_m"))
    draw_gates(ax_overlay)
    ax_overlay.set_title("实飞与 SITL 轨迹叠加（同一坐标系）")
    ax_overlay.set_xlabel("X 位置 (m)")
    ax_overlay.set_ylabel("Y 位置 (m)")
    ax_overlay.set_xlim(*xlim)
    ax_overlay.set_ylim(*ylim)
    ax_overlay.set_aspect("equal", adjustable="box")
    if datasets[0].anomaly_interval:
        ax_overlay.legend(fontsize=8, ncol=3, loc="lower left")
    else:
        ax_overlay.legend(fontsize=9, ncol=4, loc="upper right")

    for ax, ds in zip(side_axes, datasets):
        d = downsample(ds.selected, max_points)
        sc = ax.scatter(d["pos_x_m"], d["pos_y_m"], c=d["segment_time_s"], s=6, cmap="viridis", alpha=0.86)
        ax.plot(d["pos_x_m"], d["pos_y_m"], color=ds.color, lw=0.75, alpha=0.7)
        draw_yaw_arrows(ax, d, ds.color, max_arrows=18)
        draw_gates(ax)
        ax.scatter(d["pos_x_m"].iloc[0], d["pos_y_m"].iloc[0], s=65, color="#2ca02c", marker="o", label="起点")
        ax.scatter(d["pos_x_m"].iloc[-1], d["pos_y_m"].iloc[-1], s=75, color="#d62728", marker="X", label="终点")
        if ds.anomaly_interval:
            mark_trajectory_interval(ax, ds.selected, ds.anomaly_interval, ("pos_x_m", "pos_y_m"))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X 位置 (m)")
        ax.set_ylabel("Y 位置 (m)")
        ax.set_title(f"{ds.name}：轨迹与机头方向")
        if datasets[0].anomaly_interval:
            ax.legend(fontsize=7, loc="lower left", ncol=2)
        else:
            ax.legend(fontsize=8, loc="upper right")
        fig.colorbar(sc, ax=ax, label="段内时间 (s)")

    fig.suptitle("01 轨迹、门点与机头方向对比", fontsize=16, fontweight="bold")
    finish_annotated_layout(fig, datasets[0].anomaly_interval)
    path = out / "01_trajectory_comparison_cn.png"
    save_figure(fig, path, dpi=180)
    plt.close(fig)
    return path


def plot_speed_height(datasets: list[Dataset], out: Path, max_points: int, x_max_s: float) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=True)
    for ds in datasets:
        d = downsample(trim_time_window(ds.selected, x_max_s), max_points)
        axes[0].plot(d["segment_time_s"], d["speed_xy_mps"], color=ds.color, lw=1.1, label=f"{ds.name} 实际")
        if "speed_sp_xy_mps" in d:
            axes[0].plot(d["segment_time_s"], d["speed_sp_xy_mps"], color=ds.color, lw=0.9, ls="--", alpha=0.7, label=f"{ds.name} 设定")
        axes[1].plot(d["segment_time_s"], d["height_m"], color=ds.color, lw=1.1, label=f"{ds.name} 实际")
        if "height_sp_m" in d:
            axes[1].plot(d["segment_time_s"], d["height_sp_m"], color=ds.color, lw=0.9, ls="--", alpha=0.7, label=f"{ds.name} 设定")
    axes[0].set_ylabel("水平速度 (m/s)")
    axes[0].set_title("水平速度")
    axes[1].set_ylabel("高度 (m)")
    axes[1].set_xlabel("段内时间 (s)")
    axes[1].set_title("高度")
    for ax in axes:
        ax.set_xlim(0, x_max_s)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("02 速度与高度对比（真实时间刻度）", fontsize=15, fontweight="bold")
    annotate_comparison_timeseries(fig, axes, datasets)
    path = out / "02_speed_height_comparison_cn.png"
    save_figure(fig, path, dpi=180)
    plt.close(fig)
    return path


def plot_attitude(datasets: list[Dataset], out: Path, max_points: int, x_max_s: float) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10), sharex=True)
    signals = [("roll_deg", "横滚 Roll (deg)"), ("pitch_deg", "俯仰 Pitch (deg)"), ("yaw_unwrapped_deg", "展开航向 Yaw (deg)")]
    for ax, (col, ylabel) in zip(axes, signals):
        for ds in datasets:
            d = downsample(trim_time_window(ds.selected, x_max_s), max_points)
            ax.plot(d["segment_time_s"], d[col], color=ds.color, lw=1.0, label=ds.name)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" ")[0])
        ax.set_xlim(0, x_max_s)
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("段内时间 (s)")
    fig.suptitle("03 姿态角对比", fontsize=15, fontweight="bold")
    annotate_comparison_timeseries(fig, axes, datasets)
    path = out / "03_attitude_comparison_cn.png"
    save_figure(fig, path, dpi=180)
    plt.close(fig)
    return path


def plot_tracking_errors(datasets: list[Dataset], out: Path, max_points: int, x_max_s: float) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.5), sharex=True)
    plotted_any = False
    for ds in datasets:
        d = downsample(trim_time_window(ds.selected, x_max_s), max_points)
        if "pos_err_xy_m" in d:
            axes[0].plot(d["segment_time_s"], d["pos_err_xy_m"], color=ds.color, lw=1.0, label=ds.name)
            plotted_any = True
        if "height_err_m" in d:
            axes[1].plot(d["segment_time_s"], d["height_err_m"], color=ds.color, lw=1.0, label=ds.name)
            plotted_any = True
        if "yaw_err_deg" in d:
            axes[2].plot(d["segment_time_s"], np.abs(d["yaw_err_deg"]), color=ds.color, lw=1.0, label=ds.name)
            plotted_any = True
    if not plotted_any:
        axes[0].text(0.02, 0.55, "没有可用的 posSp / quatSp 设定值，无法绘制跟踪误差。", transform=axes[0].transAxes)
    axes[0].set_ylabel("XY 位置误差 (m)")
    axes[0].set_title("XY 位置跟踪误差")
    axes[1].set_ylabel("高度误差 (m)")
    axes[1].set_title("高度跟踪误差")
    axes[2].set_ylabel("|航向误差| (deg)")
    axes[2].set_title("航向跟踪误差")
    axes[2].set_xlabel("段内时间 (s)")
    for ax in axes:
        ax.set_xlim(0, x_max_s)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("04 设定值跟踪误差对比", fontsize=15, fontweight="bold")
    annotate_comparison_timeseries(fig, axes, datasets)
    path = out / "04_tracking_error_comparison_cn.png"
    save_figure(fig, path, dpi=180)
    plt.close(fig)
    return path


def plot_motor_overview(datasets: list[Dataset], out: Path, max_points: int, x_max_s: float) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10), sharex=True)
    specs = [
        ("motor_max", "最大电机指令", "归一化指令"),
        ("omega_mean_rpm", "四电机平均转速", "RPM"),
        ("omega_spread_rpm", "四电机转速离散度", "RPM"),
    ]
    for ax, (col, title, ylabel) in zip(axes, specs):
        for ds in datasets:
            if col in ds.selected:
                d = downsample(trim_time_window(ds.selected, x_max_s), max_points)
                ax.plot(d["segment_time_s"], d[col], color=ds.color, lw=1.0, label=ds.name)
        if col == "motor_max":
            ax.axhline(CMD_SATURATION, color="#555555", lw=0.9, ls="--", label="0.95 饱和线")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("段内时间 (s)")
    for ax in axes:
        ax.set_xlim(0, x_max_s)
    fig.suptitle("05 电机指令与转速总体对比", fontsize=15, fontweight="bold")
    annotate_comparison_timeseries(fig, axes, datasets)
    path = out / "05_motor_overview_comparison_cn.png"
    save_figure(fig, path, dpi=180)
    plt.close(fig)
    return path


def plot_motor_channels(datasets: list[Dataset], out: Path, max_points: int, x_max_s: float) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), sharex=True)
    for idx in range(4):
        for ds in datasets:
            d = downsample(trim_time_window(ds.selected, x_max_s), max_points)
            if f"motor_{idx}" in d:
                axes[idx, 0].plot(d["segment_time_s"], d[f"motor_{idx}"], color=ds.color, lw=0.9, label=ds.name)
            if f"omega_{idx}_rpm" in d:
                axes[idx, 1].plot(d["segment_time_s"], d[f"omega_{idx}_rpm"], color=ds.color, lw=0.9, label=ds.name)
        axes[idx, 0].set_ylabel(f"M{idx} 指令")
        axes[idx, 1].set_ylabel(f"M{idx} 转速 RPM")
        axes[idx, 0].set_title(f"电机 {idx} 指令")
        axes[idx, 1].set_title(f"电机 {idx} 转速")
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 1].legend(fontsize=8)
    axes[-1, 0].set_xlabel("段内时间 (s)")
    axes[-1, 1].set_xlabel("段内时间 (s)")
    for ax in axes.ravel():
        ax.set_xlim(0, x_max_s)
    fig.suptitle("06 四电机通道逐项对比（真实时间刻度）", fontsize=15, fontweight="bold")
    annotate_comparison_timeseries(fig, axes, datasets)
    path = out / "06_motor_channels_comparison_cn.png"
    save_figure(fig, path, dpi=180)
    plt.close(fig)
    return path


def annotate_comparison_timeseries(fig, axes, datasets, original_time=False):
    real = next(ds for ds in datasets if ds.domain == "real")
    origin = 0.0 if original_time else real.profile["selected_start_s"]
    if real.anomaly_interval:
        mark_time_interval(axes, real.anomaly_interval, origin)
    finish_annotated_layout(fig, real.anomaly_interval, origin, "NN 接管后")


def plot_summary_dashboard(summary: pd.DataFrame, out: Path, interval: dict | None = None) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    labels = summary["dataset"].tolist()
    charts = [
        ("path_xy_m", "XY 路径长度 (m)"),
        ("speed_p95_mps", "水平速度 P95 (m/s)"),
        ("position_xy_rmse_m", "XY 跟踪 RMSE (m)"),
        ("yaw_velocity_abs_p50_deg", "航向-速度夹角 P50 (deg)"),
        ("motor_cmd_max_p95", "最大电机指令 P95"),
        ("motor_any_high_sat_rate", "任一电机 >=0.95 比例"),
    ]
    for ax, (col, title) in zip(axes.ravel(), charts):
        vals = pd.to_numeric(summary[col], errors="coerce") if col in summary else pd.Series([np.nan] * len(summary))
        ax.bar(labels, vals, color=["#1f77b4", "#d62728"][: len(labels)])
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=12)
        ax.grid(True, axis="y", alpha=0.28)
        if col == "motor_any_high_sat_rate":
            ax.set_ylim(0, max(0.05, float(np.nanmax(vals)) * 1.25 if np.isfinite(vals).any() else 0.05))
            ax.yaxis.set_major_formatter(lambda x, _: f"{x * 100:.1f}%")
    fig.suptitle("07 核心指标汇总对比", fontsize=15, fontweight="bold")
    finish_annotated_layout(fig, interval, aggregate=True)
    path = out / "07_summary_dashboard_cn.png"
    save_figure(fig, path, dpi=180)
    plt.close(fig)
    return path


def plot_real_mode_timeline(real: Dataset, out: Path, max_points: int) -> Path:
    d = downsample(real.full, max_points)
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9), sharex=True)
    if "flightModeFlags" in d:
        axes[0].step(d["time_s"], d["flightModeFlags"], where="post", color="#1f77b4", lw=1.0)
    axes[0].axvspan(real.profile["selected_start_s"], real.profile["selected_end_s"], color="#f28e2b", alpha=0.22, label="分析段")
    axes[0].set_ylabel("flightModeFlags")
    axes[0].set_title("实飞模式标志")
    axes[0].legend(fontsize=8)

    axes[1].plot(d["time_s"], d["speed_xy_mps"], color="#2ca02c", lw=1.0)
    axes[1].axvspan(real.profile["selected_start_s"], real.profile["selected_end_s"], color="#f28e2b", alpha=0.22)
    axes[1].set_ylabel("水平速度 (m/s)")
    axes[1].set_title("实飞速度")

    if "motor_max" in d:
        axes[2].plot(d["time_s"], d["motor_max"], color="#d62728", lw=1.0)
        axes[2].axhline(CMD_SATURATION, color="#555555", lw=0.9, ls="--", label="0.95 饱和线")
    axes[2].axvspan(real.profile["selected_start_s"], real.profile["selected_end_s"], color="#f28e2b", alpha=0.22)
    axes[2].set_ylabel("最大电机指令")
    axes[2].set_xlabel("日志时间 (s)")
    axes[2].set_title("实飞电机指令")
    axes[2].legend(fontsize=8)

    fig.suptitle("08 实飞日志模式分段与分析窗口", fontsize=15, fontweight="bold")
    annotate_comparison_timeseries(fig, axes, [real], original_time=True)
    path = out / "08_real_mode_timeline_cn.png"
    save_figure(fig, path, dpi=180)
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, cols: list[str], rename: dict[str, str] | None = None, floatfmt: str = ".3f") -> str:
    d = df[cols].copy()
    if rename:
        d = d.rename(columns=rename)
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    lines = [
        "| " + " | ".join(str(c) for c in d.columns) + " |",
        "| " + " | ".join(["---"] * len(d.columns)) + " |",
    ]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in d.columns) + " |")
    return "\n".join(lines)


def write_method_doc(
    paths: OutputPaths,
    datasets: list[Dataset],
    summary: pd.DataFrame,
    gate_metrics: pd.DataFrame,
    motor_metrics: pd.DataFrame,
    figure_paths: list[Path],
    compare_window_s: float,
) -> None:
    profile_rows = []
    for ds in datasets:
        profile_rows.append(
            {
                "数据集": ds.name,
                "类型": "实飞" if ds.domain == "real" else "SITL",
                "原始文件": str(ds.path.relative_to(ROOT) if ds.path.is_relative_to(ROOT) else ds.path),
                "原始时长s": ds.profile["duration_s"],
                "分析起点s": ds.profile["selected_start_s"],
                "分析终点s": ds.profile["selected_end_s"],
                "分析时长s": ds.profile["selected_duration_s"],
                "采样Hz": ds.profile["approx_hz"],
                "分段口径": ds.segment_note,
            }
        )
    profile_df = pd.DataFrame(profile_rows)

    key_cols = [
        "dataset",
        "selected_duration_s",
        "path_xy_m",
        "speed_p95_mps",
        "position_xy_rmse_m",
        "yaw_velocity_abs_p50_deg",
        "motor_cmd_max_p95",
        "motor_any_high_sat_rate",
        "gate_min_distance_median",
    ]
    rename = {
        "dataset": "数据集",
        "selected_duration_s": "分析时长s",
        "path_xy_m": "XY路径m",
        "speed_p95_mps": "速度P95",
        "position_xy_rmse_m": "XY跟踪RMSE",
        "yaw_velocity_abs_p50_deg": "航向-速度P50",
        "motor_cmd_max_p95": "电机指令P95",
        "motor_any_high_sat_rate": "饱和比例",
        "gate_min_distance_median": "门点最近距离中位数",
    }

    real = summary[summary["domain"] == "real"].iloc[0]
    sitl = summary[summary["domain"] == "sitl"].iloc[0]

    lines = [
        f"# {datasets[0].name} 与 {datasets[1].name} 联合日志分析说明",
        "",
        "## 1. 数据源与分析窗口",
        "",
        markdown_table(profile_df, list(profile_df.columns), floatfmt=".2f"),
        "",
        "本次分析使用 `scripts/compare_real_sitl_logs.py` 对两份日志做同口径对比。实飞日志按神经网络控制相关的 `flightModeFlags` 截取；SITL 日志按速度和离起点位移自动截取运动段。",
        "",
        f"所有时序对比图均使用真实段内时间，横坐标单位为秒，当前对比窗口为 0-{compare_window_s:.2f} s。这个窗口默认覆盖完整较短分析段，并截取另一份日志的相同时长片段，因此可以直接比较波动频率和趋势节奏；完整截取后的时序仍保存在 `processed/` 目录。",
        "",
        "## 2. 日志处理流程",
        "",
        "1. 读取 CSV 表头，只加载分析需要的时间、位置、速度、姿态、设定值、电机指令、转速、模式标志等列。",
        "2. 将 `timeUs` 统一换算为秒，并在分析窗口内重新从 0 计时。时序图使用真实段内时间和同一秒刻度，便于比较波动频率；内部保留 `norm_time` 字段仅用于必要的二次分析。",
        "3. SITL 优先使用 `extPos[0..2]` 和 `extVel[0..2]`；实飞优先使用 `pos[0..2]` 和 `vel[0..2]`。实际来源见上方数据表；如果高度方向为 NED 负值，则用 `height=-pos_z` 作为高度。",
        "4. 姿态优先使用 `quat[0..3]` 转欧拉角，航向角展开后用于累计旋转量；若存在 `quatSp`，同步计算航向设定误差。",
        "5. 对 `posSp`、`velSp`、`quatSp` 计算位置、速度和航向跟踪误差；对 `motor[0..3]` 与 `omegaUnfiltered[0..3]` 计算指令/转速统计、相关性、估计滞后和饱和比例。",
        f"6. 使用配置中的 {len(GATES)} 个门点评估最近过门距离和局部通过次数。",
        "",
        "## 3. 核心结果摘要",
        "",
        markdown_table(summary, key_cols, rename=rename, floatfmt=".3f"),
        "",
        f"- 实飞分析段为 {real['selected_duration_s']:.2f} s，约占原始日志的 {float(real['rows_selected']) / float(real['rows_raw']):.1%}；主要来自 flags {real['flight_mode_counts']}。",
        f"- SITL 分析段为 {sitl['selected_duration_s']:.2f} s，原始日志前段存在静止等待，因此自动截取运动窗口后再统计。",
        f"- 实飞 XY 路径长度约 {real['path_xy_m']:.2f} m，SITL 约 {sitl['path_xy_m']:.2f} m；二者时长不同，时序图按真实秒刻度截取共同窗口展示，可比较频率和节奏，但不能直接当作逐点同步误差。",
        f"- 实飞最大电机指令 P95 为 {real['motor_cmd_max_p95']:.3f}，SITL 为 {sitl['motor_cmd_max_p95']:.3f}；任一电机超过 0.95 的比例分别为 {real['motor_any_high_sat_rate']:.3%} 和 {sitl['motor_any_high_sat_rate']:.3%}。",
        f"- 实飞航向与速度方向夹角 P50 为 {real['yaw_velocity_abs_p50_deg']:.2f} deg，SITL 为 {sitl['yaw_velocity_abs_p50_deg']:.2f} deg；该指标用于观察机头方向是否稳定沿运动方向或存在固定偏置。",
        "",
        "## 4. 图表输出",
        "",
    ]
    for figure in figure_paths:
        lines.append(f"- `{figure.relative_to(paths.out_dir)}`")

    lines.extend(
        [
            "",
            "## 5. 输出表格",
            "",
            "- `tables/source_profile.csv`：原始日志、分析窗口、采样率和分段口径。",
            "- `tables/comparison_metrics.csv`：两份日志的总体指标。",
            "- `tables/gate_metrics.csv`：四个门点的最近距离和通过次数。",
            "- `tables/motor_metrics.csv`：每个电机的指令、转速、相关性、滞后和饱和统计。",
            "- `processed/*_normalized.csv`：已截取分析窗口并统一字段命名后的时序数据。",
            "",
            "## 6. 注意事项",
            "",
            "- SITL 的 `flightModeFlags` 全程为 0，本报告无法用模式 bit 判断神经网络开始/结束，只能用运动段近似。",
            "- 实飞和 SITL 时长、圈数和起止状态不同，因此并排图重在分布、形态和指标对比，不代表逐时刻一一对应。",
            "- 时序图没有做时间归一化；横坐标使用秒，且两条曲线使用同一刻度尺。若要看更长 SITL 片段，可用 `--compare-window-s 20` 或更大数值重跑。",
            "- `posSp/quatSp` 是日志内的控制设定值；若控制器内部还有未记录的目标门点或网络输出，本报告只能从已记录字段间接分析。",
            "- 若后续要做严格 sim-to-real 逐圈误差，需要在日志里增加当前目标门编号、目标 yaw、神经网络原始输出和 mixer 前控制量。",
        ]
    )
    paths.doc_path.write_text("\n".join(lines), encoding="utf-8")


def make_output_paths(out_root: Path, out_name: str) -> OutputPaths:
    out_dir = out_root / out_name
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    processed_dir = out_dir / "processed"
    for path in (figures_dir, tables_dir, processed_dir):
        path.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        out_dir=out_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        processed_dir=processed_dir,
        doc_path=out_dir / "log_analysis_method_cn.md",
        manifest_path=out_dir / "deliverable_manifest.csv",
    )


def write_outputs(
    paths: OutputPaths,
    datasets: list[Dataset],
    summary: pd.DataFrame,
    gates: pd.DataFrame,
    motors: pd.DataFrame,
    passages: pd.DataFrame,
) -> list[Path]:
    profile = pd.DataFrame(
        [
            {
                "dataset": ds.name,
                "domain": ds.domain,
                "source": str(ds.path),
                "rows_raw": ds.profile["rows_raw"],
                "rows_selected": ds.profile["rows_selected"],
                "duration_s": ds.profile["duration_s"],
                "selected_start_s": ds.profile["selected_start_s"],
                "selected_end_s": ds.profile["selected_end_s"],
                "selected_duration_s": ds.profile["selected_duration_s"],
                "approx_hz": ds.profile["approx_hz"],
                "segment_note": ds.segment_note,
                "position_columns": " ".join(ds.profile.get("position_columns", [])),
                "velocity_columns": " ".join(ds.profile.get("velocity_columns", [])),
                "attitude_source": " ".join(ds.profile.get("attitude_source", [])),
                "height_convention": ds.profile.get("height_convention", ""),
                "selected_flight_mode_counts": json.dumps(ds.profile.get("selected_flight_mode_counts", {}), ensure_ascii=False),
            }
            for ds in datasets
        ]
    )
    profile.to_csv(paths.tables_dir / "source_profile.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths.tables_dir / "comparison_metrics.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(paths.tables_dir / "gate_metrics.csv", index=False, encoding="utf-8-sig")
    motors.to_csv(paths.tables_dir / "motor_metrics.csv", index=False, encoding="utf-8-sig")
    passages.to_csv(paths.tables_dir / "gate_passage_3d.csv", index=False, encoding="utf-8-sig")

    processed_paths = []
    for ds in datasets:
        path = paths.processed_dir / f"{ds.key}_normalized.csv"
        ds.selected.to_csv(path, index=False, encoding="utf-8-sig")
        processed_paths.append(path)
    return processed_paths


def write_manifest(paths: OutputPaths, figure_paths: list[Path], processed_paths: list[Path]) -> None:
    rows = []
    for path in sorted(paths.tables_dir.glob("*.csv")):
        rows.append({"type": "table", "path": str(path.relative_to(paths.out_dir))})
    for path in processed_paths:
        rows.append({"type": "processed", "path": str(path.relative_to(paths.out_dir))})
    for path in figure_paths:
        rows.append({"type": "figure", "path": str(path.relative_to(paths.out_dir))})
    rows.append({"type": "document", "path": str(paths.doc_path.relative_to(paths.out_dir))})
    pd.DataFrame(rows).to_csv(paths.manifest_path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare one real-flight CSV with one SITL CSV using the shared A2RL log-analysis workflow.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON config path. Edit this file for normal use.")
    ap.add_argument("--real-log", default=None, help="Real flight CSV path. Overrides config.real_log.")
    ap.add_argument("--sitl-log", default=None, help="SITL CSV path. Overrides config.sitl_log.")
    ap.add_argument("--out-root", default=None, help="Output root directory. Overrides config.out_root.")
    ap.add_argument("--out-name", default=None, help="Output folder name under out-root. Overrides config.out_name.")
    ap.add_argument("--real-label", default=None, help="Legend/report label for the real flight dataset.")
    ap.add_argument("--sitl-label", default=None, help="Legend/report label for the SITL dataset.")
    ap.add_argument("--real-flags", default=None, help="Comma-separated real flight control flags. Overrides config.real_flags.")
    ap.add_argument("--sitl-motion-speed", type=float, default=None, help="Speed threshold for SITL motion segment.")
    ap.add_argument("--sitl-motion-displacement", type=float, default=None, help="Displacement threshold for SITL motion segment.")
    ap.add_argument(
        "--compare-window-s",
        type=float,
        default=None,
        help="Time window in seconds for time-series comparison plots. Default: cover the shorter selected segment with a clean rounded x-axis.",
    )
    ap.add_argument("--max-plot-points", type=int, default=None, help="Maximum plotted points per dataset.")
    return ap.parse_args()


def main() -> int:
    configure_matplotlib()
    args = parse_args()
    config_path = resolve_workspace_path(args.config) if args.config else None
    config = load_config(config_path)

    real_path = resolve_workspace_path(cfg_value(config, "real_log", args.real_log, DEFAULT_REAL)).resolve()
    sitl_path = resolve_workspace_path(cfg_value(config, "sitl_log", args.sitl_log, DEFAULT_SITL)).resolve()
    out_root = resolve_workspace_path(cfg_value(config, "out_root", args.out_root, DEFAULT_OUT_ROOT)).resolve()
    out_name = str(cfg_value(config, "out_name", args.out_name, DEFAULT_OUT_NAME))
    real_label = str(cfg_value(config, "real_label", args.real_label, "实飞 circle_0704"))
    sitl_label = str(cfg_value(config, "sitl_label", args.sitl_label, "SITL 0628"))
    real_key = str(config.get("real_key", Path(real_path).stem.replace(".", "_")))
    sitl_key = str(config.get("sitl_key", Path(sitl_path).stem.replace(".", "_")))
    real_flags = parse_flags_value(cfg_value(config, "real_flags", args.real_flags, REAL_CONTROL_FLAGS))
    sitl_motion_speed = float(cfg_value(config, "sitl_motion_speed", args.sitl_motion_speed, SITL_MOTION_SPEED_MPS))
    sitl_motion_displacement = float(
        cfg_value(config, "sitl_motion_displacement", args.sitl_motion_displacement, SITL_MOTION_DISPLACEMENT_M)
    )
    compare_window_arg = cfg_value(config, "compare_window_s", args.compare_window_s, None)
    compare_window_arg = None if compare_window_arg in (None, "") else float(compare_window_arg)
    max_plot_points = int(cfg_value(config, "max_plot_points", args.max_plot_points, MAX_PLOT_POINTS))

    global GATES, GATE_LABELS
    if "gates" in config:
        raw_gates = np.asarray(config["gates"], dtype=float)
        gate_centers = normalize_gates(raw_gates)
        GATES = raw_gates if raw_gates.shape[1] == 4 else gate_centers
    if "gate_labels" in config:
        GATE_LABELS = [str(item) for item in config["gate_labels"]]
    GATE_LABELS = normalize_gate_labels(GATE_LABELS, len(GATES))

    paths = make_output_paths(out_root, out_name)
    datasets = [
        load_dataset(
            real_key,
            real_label,
            "real",
            str(config.get("real_color", "#1f77b4")),
            real_path,
            real_flags,
            sitl_motion_speed,
            sitl_motion_displacement,
        ),
        load_dataset(
            sitl_key,
            sitl_label,
            "sitl",
            str(config.get("sitl_color", "#d62728")),
            sitl_path,
            real_flags,
            sitl_motion_speed,
            sitl_motion_displacement,
        ),
    ]

    datasets[0].anomaly_interval = config.get("real_anomaly_interval")
    gate_tables = [make_gate_metrics(ds) for ds in datasets]
    passage_tables = [evaluate_gate_passages(ds.selected, GATES, GATE_LABELS, ds.name) for ds in datasets]
    motor_tables = [make_motor_metrics(ds) for ds in datasets]
    gates = pd.concat(gate_tables, ignore_index=True)
    passages = pd.concat(passage_tables, ignore_index=True)
    motors = pd.concat(motor_tables, ignore_index=True)
    summary = pd.DataFrame([make_summary(ds, gate_table) for ds, gate_table in zip(datasets, gate_tables)])
    for ds, passage_table in zip(datasets, passage_tables):
        mask = summary["dataset"] == ds.name
        summary.loc[mask, "gates_passed_3d"] = int(passage_table["passed"].sum())
        summary.loc[mask, "gates_total_3d"] = int(len(passage_table))

    processed_paths = write_outputs(paths, datasets, summary, gates, motors, passages)
    max_points = max(100, max_plot_points)
    compare_window_s = comparison_time_limit(datasets, compare_window_arg)
    figure_paths = [
        plot_trajectory(datasets, paths.figures_dir, max_points),
        plot_speed_height(datasets, paths.figures_dir, max_points, compare_window_s),
        plot_attitude(datasets, paths.figures_dir, max_points, compare_window_s),
        plot_tracking_errors(datasets, paths.figures_dir, max_points, compare_window_s),
        plot_motor_overview(datasets, paths.figures_dir, max_points, compare_window_s),
        plot_motor_channels(datasets, paths.figures_dir, max_points, compare_window_s),
        plot_summary_dashboard(summary, paths.figures_dir, datasets[0].anomaly_interval),
        plot_real_mode_timeline(datasets[0], paths.figures_dir, max_points),
        plot_trajectory_gate_3d(
            [(ds.name, ds.color, ds.selected) for ds in datasets],
            passage_tables,
            GATES,
            GATE_LABELS,
            paths.figures_dir / "09_trajectory_3d_with_gates.png",
            max_points,
            "实飞与 SITL 三维轨迹及 1.5 m × 1.5 m 门框",
            anomaly_intervals=[ds.anomaly_interval for ds in datasets],
        ),
    ]
    write_method_doc(paths, datasets, summary, gates, motors, figure_paths, compare_window_s)
    write_manifest(paths, figure_paths, processed_paths)

    print(f"Output directory: {paths.out_dir}")
    print(f"Document: {paths.doc_path}")
    if config_path and config_path.exists():
        print(f"Config: {config_path}")
    print(f"Time-series comparison window: 0-{compare_window_s:.2f} s")
    print("Generated figures:")
    for path in figure_paths:
        print(f"  {path}")
    print("Summary:")
    print(summary[["dataset", "selected_duration_s", "path_xy_m", "speed_p95_mps", "motor_cmd_max_p95", "motor_any_high_sat_rate"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
