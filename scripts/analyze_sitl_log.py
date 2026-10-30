from __future__ import annotations

# python scripts\analyze_sitl_log.py --log data\raw\sitl\LOG_0706_lyh_8zi.csv --label LOG_0706_lyh_8zi

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from track_3d import (
    DEFAULT_GATE_LABELS,
    DEFAULT_GATES_NED,
    evaluate_gate_passages,
    plot_trajectory_gate_3d,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "data" / "raw" / "sitl" / "LOG_SITL_0628lyh.csv"
DEFAULT_OUT_ROOT = ROOT / "reports"
MAX_PLOT_POINTS = 5000
# 当前赛道门定义：[x, y, z(NED), yaw(rad)]，yaw 为门平面法向在 XY 平面的朝向
# 可用 --track-config 加载 [x, y, z(NED), yaw(deg)] 赛道配置
GATES = np.array(
    [
        [4.0, 0.0, -2.0, np.pi / 2],
        [0.0, 4.0, -2.0, np.pi],
        [-4.0, 0.0, -2.0, -np.pi / 2],
        [0.0, -4.0, -2.0, 0.0],
    ],
    dtype=float,
)
GATE_LABELS = [f"Gate {idx:02d}" for idx in range(1, len(GATES) + 1)]

@dataclass
class AnalysisPaths:
    out_dir: Path
    figures_dir: Path
    tables_dir: Path
    summary_json: Path


def configure_matplotlib() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.28


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def vector_cols(columns: set[str], prefix: str, n: int = 3) -> list[str]:
    out = [f"{prefix}[{idx}]" for idx in range(n)]
    return out if all(col in columns for col in out) else []


def indexed_cols(columns: set[str], prefix: str, max_n: int = 16) -> list[str]:
    return [f"{prefix}[{idx}]" for idx in range(max_n) if f"{prefix}[{idx}]" in columns]


def first_vector(columns: set[str], prefixes: Iterable[str], n: int = 3) -> tuple[str | None, list[str]]:
    for prefix in prefixes:
        cols = vector_cols(columns, prefix, n=n)
        if cols:
            return prefix, cols
    return None, []


def wanted_columns(columns: list[str]) -> list[str]:
    available = set(columns)
    wanted = ["timeS", "timeUs", "loopIteration", "flightModeFlags", "stateFlags"]
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
        "attSp",
    ):
        n = 4 if prefix in {"quat", "quatSp"} else 3
        wanted.extend(f"{prefix}[{idx}]" for idx in range(n))
    for prefix in ("motor", "omega", "omegaUnfiltered", "omega_dot"):
        wanted.extend(f"{prefix}[{idx}]" for idx in range(16))
    return [col for col in wanted if col in available]


def load_sitl_frame(path: Path, start_s: float | None = None, end_s: float | None = None,
                    position_source: str | None = None, velocity_source: str | None = None) -> tuple[pd.DataFrame, dict]:
    columns = read_header(path)
    usecols = wanted_columns(columns)
    raw = pd.read_csv(path, usecols=usecols)
    available = set(raw.columns)

    if "timeS" in raw:
        time_s = pd.to_numeric(raw["timeS"], errors="coerce").astype(float)
        time_s = time_s - float(time_s.iloc[0])
    elif "timeUs" in raw:
        time_s = pd.to_numeric(raw["timeUs"], errors="coerce").astype(float)
        time_s = (time_s - float(time_s.iloc[0])) / 1_000_000.0
    else:
        raise ValueError("CSV needs timeS or timeUs for time-axis analysis.")

    pos_prefix, pos_cols = first_vector(available, (position_source,) if position_source else ("extPos", "pos", "ekf_pos"))
    vel_prefix, vel_cols = first_vector(available, (velocity_source,) if velocity_source else ("extVel", "vel", "ekf_vel"))
    pos_sp_prefix, pos_sp_cols = first_vector(available, ("posSp",))
    vel_sp_prefix, vel_sp_cols = first_vector(available, ("velSp",))
    _, quat_cols = first_vector(available, ("quat",), n=4)
    _, quat_sp_cols = first_vector(available, ("quatSp",), n=4)
    att_prefix, att_cols = first_vector(available, ("extAtt", "ekf_att"), n=3)
    motor_cols = indexed_cols(available, "motor")
    omega_cols = indexed_cols(available, "omega")
    omega_unfiltered_cols = indexed_cols(available, "omegaUnfiltered")
    omega_dot_cols = indexed_cols(available, "omega_dot")

    if not pos_cols:
        raise ValueError("CSV needs pos[0..2], extPos[0..2], or ekf_pos[0..2].")
    if not vel_cols:
        raise ValueError("CSV needs vel[0..2], extVel[0..2], or ekf_vel[0..2].")
    if not quat_cols and not att_cols:
        raise ValueError("CSV needs quat[0..3] or attitude columns such as extAtt[0..2].")

    df = pd.DataFrame({"time_s": time_s})
    for axis, col in zip(("x", "y", "z"), pos_cols):
        df[f"pos_{axis}_m"] = pd.to_numeric(raw[col], errors="coerce")
    for axis, col in zip(("x", "y", "z"), vel_cols):
        df[f"vel_{axis}_mps"] = pd.to_numeric(raw[col], errors="coerce")
    for axis, col in zip(("x", "y", "z"), pos_sp_cols):
        df[f"pos_sp_{axis}_m"] = pd.to_numeric(raw[col], errors="coerce")
    for axis, col in zip(("x", "y", "z"), vel_sp_cols):
        df[f"vel_sp_{axis}_mps"] = pd.to_numeric(raw[col], errors="coerce")
    for idx, col in enumerate(motor_cols):
        df[f"motor_cmd_{idx}"] = pd.to_numeric(raw[col], errors="coerce")
    for idx, col in enumerate(omega_cols):
        df[f"omega_{idx}_rpm"] = pd.to_numeric(raw[col], errors="coerce")
    for idx, col in enumerate(omega_unfiltered_cols):
        df[f"omega_unfiltered_{idx}_rpm"] = pd.to_numeric(raw[col], errors="coerce")
    for idx, col in enumerate(omega_dot_cols):
        df[f"omega_dot_{idx}"] = pd.to_numeric(raw[col], errors="coerce")

    if quat_cols:
        quat = raw[quat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        euler = quat_to_euler_rad(quat)
        df["roll_rad"] = euler[:, 0]
        df["pitch_rad"] = euler[:, 1]
        df["yaw_rad"] = wrap_pi(euler[:, 2])
    else:
        att = raw[att_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.nanmedian(np.abs(att)) > math.pi * 1.5:
            att = np.deg2rad(att)
        df["roll_rad"] = att[:, 0]
        df["pitch_rad"] = att[:, 1]
        df["yaw_rad"] = wrap_pi(att[:, 2])

    if quat_sp_cols:
        quat_sp = raw[quat_sp_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        norm = np.linalg.norm(quat_sp, axis=1)
        if np.nanmedian(norm) > 0.5:
            euler_sp = quat_to_euler_rad(quat_sp)
            df["yaw_sp_rad"] = wrap_pi(euler_sp[:, 2])
    elif "attSp[2]" in raw:
        yaw_sp = pd.to_numeric(raw["attSp[2]"], errors="coerce").to_numpy(dtype=float)
        if np.nanmedian(np.abs(yaw_sp)) > math.pi * 1.5:
            yaw_sp = np.deg2rad(yaw_sp)
        df["yaw_sp_rad"] = wrap_pi(yaw_sp)

    z_reference = df["pos_z_m"]
    if "pos_sp_z_m" in df:
        z_reference = pd.concat([df["pos_z_m"], df["pos_sp_z_m"]], axis=0)
    ned_z_down = float(np.nanmedian(z_reference)) < 0.0
    z_sign = -1.0 if ned_z_down else 1.0
    df["height_m"] = z_sign * df["pos_z_m"]
    df["speed_xy_mps"] = np.hypot(df["vel_x_mps"], df["vel_y_mps"])
    df["speed_3d_mps"] = np.sqrt(df["vel_x_mps"] ** 2 + df["vel_y_mps"] ** 2 + df["vel_z_mps"] ** 2)
    df["yaw_deg"] = np.rad2deg(df["yaw_rad"])
    df["yaw_unwrapped_deg"] = np.rad2deg(np.unwrap(df["yaw_rad"].to_numpy(dtype=float)))

    if all(col in df for col in ("pos_sp_x_m", "pos_sp_y_m", "pos_sp_z_m")):
        df["height_sp_m"] = z_sign * df["pos_sp_z_m"]
        df["err_x_m"] = df["pos_x_m"] - df["pos_sp_x_m"]
        df["err_y_m"] = df["pos_y_m"] - df["pos_sp_y_m"]
        df["err_height_m"] = df["height_m"] - df["height_sp_m"]
        df["err_xy_m"] = np.hypot(df["err_x_m"], df["err_y_m"])
        df["err_3d_m"] = np.sqrt(df["err_xy_m"] ** 2 + df["err_height_m"] ** 2)
    if all(col in df for col in ("vel_sp_x_mps", "vel_sp_y_mps", "vel_sp_z_mps")):
        df["speed_sp_xy_mps"] = np.hypot(df["vel_sp_x_mps"], df["vel_sp_y_mps"])
        df["speed_sp_3d_mps"] = np.sqrt(
            df["vel_sp_x_mps"] ** 2 + df["vel_sp_y_mps"] ** 2 + df["vel_sp_z_mps"] ** 2
        )
        df["speed_xy_err_mps"] = df["speed_xy_mps"] - df["speed_sp_xy_mps"]
    if "yaw_sp_rad" in df:
        df["yaw_sp_deg"] = np.rad2deg(df["yaw_sp_rad"])
        df["yaw_err_deg"] = np.rad2deg(wrap_pi(df["yaw_rad"].to_numpy() - df["yaw_sp_rad"].to_numpy()))

    motor_indices = sorted(
        {
            int(col.split("[", 1)[1].split("]", 1)[0])
            for cols in (motor_cols, omega_cols)
            for col in cols
        }
    )
    active_motor_indices: list[int] = []
    for idx in motor_indices:
        cmd_col = f"motor_cmd_{idx}"
        rpm_col = f"omega_{idx}_rpm"
        cmd = df[cmd_col].to_numpy(dtype=float) if cmd_col in df else np.array([])
        rpm = df[rpm_col].to_numpy(dtype=float) if rpm_col in df else np.array([])
        cmd_std = float(np.nanstd(cmd)) if len(cmd) else 0.0
        rpm_p95 = float(np.nanpercentile(rpm, 95)) if len(rpm) else 0.0
        rpm_std = float(np.nanstd(rpm)) if len(rpm) else 0.0
        if rpm_p95 > 1.0 or rpm_std > 1.0 or cmd_std > 1e-3:
            active_motor_indices.append(idx)
    active_motor_indices = [idx for idx in active_motor_indices if f"omega_{idx}_rpm" in df or f"motor_cmd_{idx}" in df]

    if active_motor_indices:
        cmd_cols_norm = [f"motor_cmd_{idx}" for idx in active_motor_indices if f"motor_cmd_{idx}" in df]
        rpm_cols_norm = [f"omega_{idx}_rpm" for idx in active_motor_indices if f"omega_{idx}_rpm" in df]
        if cmd_cols_norm:
            cmd_matrix = df[cmd_cols_norm].to_numpy(dtype=float)
            df["motor_cmd_mean"] = np.nanmean(cmd_matrix, axis=1)
            df["motor_cmd_spread"] = np.nanmax(cmd_matrix, axis=1) - np.nanmin(cmd_matrix, axis=1)
        if rpm_cols_norm:
            rpm_matrix = df[rpm_cols_norm].to_numpy(dtype=float)
            df["omega_mean_rpm"] = np.nanmean(rpm_matrix, axis=1)
            df["omega_spread_rpm"] = np.nanmax(rpm_matrix, axis=1) - np.nanmin(rpm_matrix, axis=1)

    mask = np.ones(len(df), dtype=bool)
    if start_s is not None:
        mask &= df["time_s"].to_numpy() >= start_s
    if end_s is not None:
        mask &= df["time_s"].to_numpy() <= end_s
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        raise ValueError("Selected time window is empty.")
    df["time_s"] = df["time_s"] - float(df["time_s"].iloc[0])

    profile = {
        "source": str(path),
        "rows_raw": int(len(raw)),
        "rows_used": int(len(df)),
        "columns_raw": int(len(columns)),
        "duration_s": float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]),
        "approx_hz": float((len(df) - 1) / max(df["time_s"].iloc[-1] - df["time_s"].iloc[0], 1e-9)),
        "position_columns": pos_cols,
        "velocity_columns": vel_cols,
        "position_setpoint_columns": pos_sp_cols,
        "velocity_setpoint_columns": vel_sp_cols,
        "attitude_source": quat_cols or att_cols,
        "yaw_setpoint_source": quat_sp_cols or (["attSp[2]"] if "attSp[2]" in raw else []),
        "motor_columns": motor_cols,
        "omega_columns": omega_cols,
        "omega_unfiltered_columns": omega_unfiltered_cols,
        "omega_dot_columns": omega_dot_cols,
        "active_motor_indices": active_motor_indices,
        "height_convention": "NED z down, height = -z" if ned_z_down else "z up, height = z",
        "time_window_s": [start_s, end_s],
    }
    if pos_prefix:
        profile["position_prefix"] = pos_prefix
    if vel_prefix:
        profile["velocity_prefix"] = vel_prefix
    if pos_sp_prefix:
        profile["position_setpoint_prefix"] = pos_sp_prefix
    if vel_sp_prefix:
        profile["velocity_setpoint_prefix"] = vel_sp_prefix
    if att_prefix:
        profile["attitude_prefix"] = att_prefix
    return df, profile


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


def corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 10:
        return float("nan")
    aa = aa[mask]
    bb = bb[mask]
    if np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def best_lag_ms(cmd: pd.Series | np.ndarray, rpm: pd.Series | np.ndarray, dt_s: float, max_lag_ms: float = 300.0) -> float:
    cmd_arr = np.asarray(cmd, dtype=float)
    rpm_arr = np.asarray(rpm, dtype=float)
    mask = np.isfinite(cmd_arr) & np.isfinite(rpm_arr)
    cmd_arr = cmd_arr[mask]
    rpm_arr = rpm_arr[mask]
    if len(cmd_arr) < 30 or dt_s <= 0:
        return float("nan")
    cmd_arr = cmd_arr - np.mean(cmd_arr)
    rpm_arr = rpm_arr - np.mean(rpm_arr)
    if np.std(cmd_arr) == 0.0 or np.std(rpm_arr) == 0.0:
        return float("nan")
    max_lag = max(1, int(max_lag_ms / (dt_s * 1000.0)))
    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = cmd_arr[-lag:]
            y = rpm_arr[:lag]
        elif lag > 0:
            x = cmd_arr[:-lag]
            y = rpm_arr[lag:]
        else:
            x = cmd_arr
            y = rpm_arr
        if len(x) < 20:
            continue
        score = corr(x, y)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_lag = lag
    return float(best_lag * dt_s * 1000.0)


def make_metrics(df: pd.DataFrame, profile: dict) -> tuple[pd.DataFrame, dict]:
    dt = np.diff(df["time_s"].to_numpy(dtype=float))
    median_dt_s = float(np.nanmedian(dt)) if len(dt) else float("nan")
    dpos = np.diff(df[["pos_x_m", "pos_y_m", "pos_z_m"]].to_numpy(dtype=float), axis=0)
    path_len_3d = float(np.nansum(np.linalg.norm(dpos, axis=1))) if len(dpos) else float("nan")
    path_len_xy = float(np.nansum(np.linalg.norm(dpos[:, :2], axis=1))) if len(dpos) else float("nan")

    rows: list[dict[str, str | float]] = []

    def add_group(group: str, signal: str, unit: str, values: pd.Series | np.ndarray) -> None:
        row = {"group": group, "signal": signal, "unit": unit}
        row.update(stats(values))
        rows.append(row)

    add_group("speed", "horizontal speed", "m/s", df["speed_xy_mps"])
    add_group("speed", "3D speed", "m/s", df["speed_3d_mps"])
    add_group("height", "height", "m", df["height_m"])
    add_group("yaw", "yaw wrapped", "deg", df["yaw_deg"])
    add_group("yaw", "yaw unwrapped", "deg", df["yaw_unwrapped_deg"])
    if "speed_xy_err_mps" in df:
        add_group("tracking", "horizontal speed error", "m/s", df["speed_xy_err_mps"])
    if "err_xy_m" in df:
        add_group("tracking", "horizontal position error", "m", df["err_xy_m"])
        add_group("tracking", "height error", "m", df["err_height_m"])
        add_group("tracking", "3D position error", "m", df["err_3d_m"])
    if "yaw_err_deg" in df:
        add_group("tracking", "yaw error", "deg", df["yaw_err_deg"])

    active_motor_indices = list(profile.get("active_motor_indices", []))
    motor_corrs: list[float] = []
    motor_lags: list[float] = []
    for idx in active_motor_indices:
        cmd_col = f"motor_cmd_{idx}"
        rpm_col = f"omega_{idx}_rpm"
        omega_dot_col = f"omega_dot_{idx}"
        if cmd_col in df:
            add_group("motor_command", f"motor[{idx}] command", "normalized", df[cmd_col])
        if rpm_col in df:
            add_group("motor_omega", f"omega[{idx}]", "rpm", df[rpm_col])
        if omega_dot_col in df:
            add_group("motor_omega_dot", f"omega_dot[{idx}]", "rpm/s", df[omega_dot_col])
        if cmd_col in df and rpm_col in df:
            c = corr(df[cmd_col], df[rpm_col])
            lag = best_lag_ms(df[cmd_col], df[rpm_col], median_dt_s)
            motor_corrs.append(c)
            motor_lags.append(lag)
            rows.append(
                {
                    "group": "motor_response",
                    "signal": f"motor[{idx}] command vs omega",
                    "unit": "",
                    "mean": c,
                    "std": float("nan"),
                    "min": float("nan"),
                    "p05": float("nan"),
                    "p50": lag,
                    "p95": float("nan"),
                    "max": float("nan"),
                    "motor_index": idx,
                    "cmd_rpm_corr": c,
                    "best_lag_ms": lag,
                    "cmd_high_sat_rate": float(np.mean(df[cmd_col].to_numpy(dtype=float) >= 0.95)),
                    "cmd_low_sat_rate": float(np.mean(df[cmd_col].to_numpy(dtype=float) <= 0.05)),
                }
            )
    if "motor_cmd_mean" in df:
        add_group("motor_command", "active motor command mean", "normalized", df["motor_cmd_mean"])
        add_group("motor_command", "active motor command spread", "normalized", df["motor_cmd_spread"])
    if "omega_mean_rpm" in df:
        add_group("motor_omega", "active omega mean", "rpm", df["omega_mean_rpm"])
        add_group("motor_omega", "active omega spread", "rpm", df["omega_spread_rpm"])

    summary = {
        **profile,
        "path_length_xy_m": path_len_xy,
        "path_length_3d_m": path_len_3d,
        "x_range_m": float(df["pos_x_m"].max() - df["pos_x_m"].min()),
        "y_range_m": float(df["pos_y_m"].max() - df["pos_y_m"].min()),
        "height_range_m": float(df["height_m"].max() - df["height_m"].min()),
        "speed_xy_mean_mps": stats(df["speed_xy_mps"])["mean"],
        "speed_xy_p95_mps": stats(df["speed_xy_mps"])["p95"],
        "speed_xy_max_mps": stats(df["speed_xy_mps"])["max"],
        "height_mean_m": stats(df["height_m"])["mean"],
        "height_min_m": stats(df["height_m"])["min"],
        "height_max_m": stats(df["height_m"])["max"],
        "yaw_total_rotation_deg": float(df["yaw_unwrapped_deg"].iloc[-1] - df["yaw_unwrapped_deg"].iloc[0]),
        "median_dt_ms": float(median_dt_s * 1000.0) if np.isfinite(median_dt_s) else float("nan"),
        "active_motor_count": len(active_motor_indices),
    }
    if "err_xy_m" in df:
        summary["position_xy_rmse_m"] = rmse(df["err_xy_m"])
        summary["height_rmse_m"] = rmse(df["err_height_m"])
        summary["position_3d_rmse_m"] = rmse(df["err_3d_m"])
    if "speed_xy_err_mps" in df:
        summary["speed_xy_rmse_mps"] = rmse(df["speed_xy_err_mps"])
    if "yaw_err_deg" in df:
        summary["yaw_rmse_deg"] = rmse(df["yaw_err_deg"])
        summary["yaw_abs_p95_deg"] = float(np.percentile(np.abs(finite(df["yaw_err_deg"])), 95))
    if "motor_cmd_mean" in df:
        summary["motor_cmd_mean"] = stats(df["motor_cmd_mean"])["mean"]
        summary["motor_cmd_p95"] = stats(df["motor_cmd_mean"])["p95"]
        summary["motor_cmd_spread_p95"] = stats(df["motor_cmd_spread"])["p95"]
    if "omega_mean_rpm" in df:
        summary["omega_mean_rpm"] = stats(df["omega_mean_rpm"])["mean"]
        summary["omega_p95_rpm"] = stats(df["omega_mean_rpm"])["p95"]
        summary["omega_max_rpm"] = stats(df["omega_mean_rpm"])["max"]
        summary["omega_spread_p95_rpm"] = stats(df["omega_spread_rpm"])["p95"]
    finite_corrs = [x for x in motor_corrs if np.isfinite(x)]
    finite_lags = [x for x in motor_lags if np.isfinite(x)]
    if finite_corrs:
        summary["motor_cmd_omega_corr_mean"] = float(np.mean(finite_corrs))
    if finite_lags:
        summary["motor_cmd_omega_lag_ms_mean"] = float(np.mean(finite_lags))

    return pd.DataFrame(rows), summary


def plot_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def plot_speed_height(df: pd.DataFrame, out: Path, max_points: int) -> Path:
    idx = plot_indices(len(df), max_points)
    d = df.iloc[idx]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(d["time_s"], d["speed_xy_mps"], label="Horizontal speed", lw=1.2)
    axes[0].plot(d["time_s"], d["speed_3d_mps"], label="3D speed", lw=1.0, alpha=0.8)
    if "speed_sp_xy_mps" in d:
        axes[0].plot(d["time_s"], d["speed_sp_xy_mps"], label="Horizontal speed setpoint", lw=1.0, ls="--")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Speed response")
    axes[0].legend(ncol=3, fontsize=9)

    axes[1].plot(d["time_s"], d["height_m"], label="Height", lw=1.2, color="#2ca02c")
    if "height_sp_m" in d:
        axes[1].plot(d["time_s"], d["height_sp_m"], label="Height setpoint", lw=1.0, ls="--", color="#111111")
    axes[1].set_ylabel("Height (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Height response")
    axes[1].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    path = out / "01_speed_height.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_trajectory(df: pd.DataFrame, out: Path, max_points: int) -> Path:
    idx = plot_indices(len(df), max_points)
    d = df.iloc[idx]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sc = axes[0].scatter(d["pos_x_m"], d["pos_y_m"], c=d["time_s"], s=8, cmap="viridis", alpha=0.9)
    axes[0].plot(d["pos_x_m"], d["pos_y_m"], color="#1f77b4", lw=0.5, alpha=0.55)
    if all(col in d for col in ("pos_sp_x_m", "pos_sp_y_m")):
        axes[0].plot(d["pos_sp_x_m"], d["pos_sp_y_m"], color="#d62728", lw=1.0, ls="--", label="Setpoint")
        axes[0].legend(fontsize=9)
    axes[0].scatter(d["pos_x_m"].iloc[0], d["pos_y_m"].iloc[0], marker="o", s=55, color="#2ca02c", label="Start")
    axes[0].scatter(d["pos_x_m"].iloc[-1], d["pos_y_m"].iloc[-1], marker="X", s=65, color="#d62728", label="End")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title("XY trajectory")
    fig.colorbar(sc, ax=axes[0], label="Time (s)")

    axes[1].plot(d["time_s"], d["pos_x_m"], label="X", lw=1.0)
    axes[1].plot(d["time_s"], d["pos_y_m"], label="Y", lw=1.0)
    axes[1].plot(d["time_s"], d["height_m"], label="Height", lw=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position / height (m)")
    axes[1].set_title("Position components")
    axes[1].legend(ncol=3, fontsize=9)
    fig.tight_layout()
    path = out / "02_trajectory_xy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_trajectory_heading_overview(df: pd.DataFrame, out: Path, max_points: int) -> Path:
    idx = plot_indices(len(df), max_points)
    d = df.iloc[idx]
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], wspace=0.18, hspace=0.32)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_yaw = fig.add_subplot(gs[1, 1])

    color = "#d62728"
    ax_xy.plot(d["pos_x_m"], d["pos_y_m"], color=color, lw=1.25, label="SITL")
    arrow_step = max(1, len(d) // 18)
    q = d.iloc[::arrow_step]
    ax_xy.quiver(
        q["pos_x_m"],
        q["pos_y_m"],
        np.cos(q["yaw_rad"]),
        np.sin(q["yaw_rad"]),
        color=color,
        alpha=0.28,
        width=0.0035,
        scale=16,
        headwidth=4.0,
        headlength=5.0,
        label="机头朝向",
    )
    ax_xy.scatter(GATES[:, 0], GATES[:, 1], c="black", s=58, marker="x", label="门点")
    for label, gate in zip(GATE_LABELS, GATES):
        gx, gy = gate[:2]
        ax_xy.text(gx + 0.05, gy + 0.05, label, fontsize=9)
    ax_xy.scatter(d["pos_x_m"].iloc[0], d["pos_y_m"].iloc[0], marker="o", s=55, color="#2ca02c", label="Start")
    ax_xy.scatter(d["pos_x_m"].iloc[-1], d["pos_y_m"].iloc[-1], marker="X", s=70, color=color, label="End")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("X 位置 (m)")
    ax_xy.set_ylabel("Y 位置 (m)")
    ax_xy.set_title("XY 轨迹")
    ax_xy.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)

    ax_h.plot(d["time_s"], d["height_m"], color=color, lw=1.1, label="SITL")
    if "height_sp_m" in d:
        ax_h.plot(d["time_s"], d["height_sp_m"], color="#111111", lw=0.9, ls="--", label="高度设定")
    ax_h.set_title("高度")
    ax_h.set_xlabel("时间 (s)")
    ax_h.set_ylabel("高度 (m)")
    ax_h.legend(fontsize=8, loc="upper right")

    ax_yaw.plot(d["time_s"], d["yaw_deg"], color=color, lw=1.1, label="SITL")
    if "yaw_sp_deg" in d:
        ax_yaw.plot(d["time_s"], d["yaw_sp_deg"], color="#ff7f0e", lw=0.9, ls="--", label="航向设定")
    ax_yaw.set_title("航向角时序")
    ax_yaw.set_xlabel("时间 (s)")
    ax_yaw.set_ylabel("航向角 (deg)")
    ax_yaw.legend(fontsize=8, loc="upper right")

    fig.suptitle("轨迹、高度与航向", fontsize=15, fontweight="bold")
    path = out / "06_trajectory_heading_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_yaw(df: pd.DataFrame, out: Path, max_points: int) -> Path:
    idx = plot_indices(len(df), max_points)
    d = df.iloc[idx]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(d["time_s"], d["yaw_deg"], label="Yaw wrapped", lw=1.0)
    if "yaw_sp_deg" in d:
        axes[0].plot(d["time_s"], d["yaw_sp_deg"], label="Yaw setpoint", lw=1.0, ls="--")
    axes[0].set_ylabel("Yaw (deg)")
    axes[0].set_title("Yaw angle")
    axes[0].legend(ncol=2, fontsize=9)

    axes[1].plot(d["time_s"], d["yaw_unwrapped_deg"], label="Yaw unwrapped", lw=1.0, color="#9467bd")
    if "yaw_err_deg" in d:
        ax2 = axes[1].twinx()
        ax2.plot(d["time_s"], d["yaw_err_deg"], label="Yaw error", lw=0.8, color="#d62728", alpha=0.75)
        ax2.set_ylabel("Yaw error (deg)")
    axes[1].set_ylabel("Unwrapped yaw (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Yaw rotation and tracking error")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = out / "03_yaw.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_errors(df: pd.DataFrame, out: Path, max_points: int) -> Path | None:
    if "err_xy_m" not in df:
        return None
    idx = plot_indices(len(df), max_points)
    d = df.iloc[idx]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(d["time_s"], d["err_x_m"], label="X error", lw=1.0)
    axes[0].plot(d["time_s"], d["err_y_m"], label="Y error", lw=1.0)
    axes[0].set_ylabel("Error (m)")
    axes[0].set_title("Horizontal position errors")
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].plot(d["time_s"], d["err_height_m"], label="Height error", lw=1.0, color="#2ca02c")
    axes[1].set_ylabel("Error (m)")
    axes[1].set_title("Height tracking error")
    axes[1].legend(fontsize=9)
    axes[2].plot(d["time_s"], d["err_xy_m"], label="XY norm", lw=1.0)
    axes[2].plot(d["time_s"], d["err_3d_m"], label="3D norm", lw=1.0)
    axes[2].set_ylabel("Error norm (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Position error norm")
    axes[2].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    path = out / "04_position_errors.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def active_motor_indices_from_frame(df: pd.DataFrame) -> list[int]:
    indices: list[int] = []
    for col in df.columns:
        if col.startswith("motor_cmd_"):
            suffix = col.removeprefix("motor_cmd_")
            if suffix.isdigit():
                indices.append(int(suffix))
    out: list[int] = []
    for idx in sorted(set(indices)):
        cmd_col = f"motor_cmd_{idx}"
        rpm_col = f"omega_{idx}_rpm"
        cmd_std = float(np.nanstd(df[cmd_col].to_numpy(dtype=float))) if cmd_col in df else 0.0
        rpm_p95 = float(np.nanpercentile(df[rpm_col].to_numpy(dtype=float), 95)) if rpm_col in df else 0.0
        rpm_std = float(np.nanstd(df[rpm_col].to_numpy(dtype=float))) if rpm_col in df else 0.0
        if rpm_p95 > 1.0 or rpm_std > 1.0 or cmd_std > 1e-3:
            out.append(idx)
    return out


def plot_motor_commands(df: pd.DataFrame, out: Path, max_points: int) -> Path | None:
    motors = active_motor_indices_from_frame(df)
    if not motors:
        return None
    idx = plot_indices(len(df), max_points)
    d = df.iloc[idx]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(motors), 4)))
    for color, motor_idx in zip(colors, motors):
        col = f"motor_cmd_{motor_idx}"
        if col in d:
            axes[0].plot(d["time_s"], d[col], lw=0.95, label=f"M{motor_idx}", color=color)
    axes[0].set_title("电机命令时序")
    axes[0].set_ylabel("命令值")
    axes[0].legend(ncol=min(len(motors), 4), fontsize=9)

    if "motor_cmd_mean" in d:
        axes[1].plot(d["time_s"], d["motor_cmd_mean"], label="平均命令", lw=1.2, color="#111111")
    if "motor_cmd_spread" in d:
        axes[1].plot(d["time_s"], d["motor_cmd_spread"], label="命令极差", lw=1.0, color="#d62728")
    axes[1].axhline(0.95, color="#d62728", lw=0.8, ls="--", alpha=0.7, label="高饱和参考 0.95")
    axes[1].set_title("电机命令平均值与离散度")
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("命令值")
    axes[1].legend(ncol=3, fontsize=9)
    fig.tight_layout()
    path = out / "07_motor_commands.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_motor_omega(df: pd.DataFrame, out: Path, max_points: int) -> Path | None:
    motors = active_motor_indices_from_frame(df)
    if not motors:
        return None
    idx = plot_indices(len(df), max_points)
    d = df.iloc[idx]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(motors), 4)))
    for color, motor_idx in zip(colors, motors):
        col = f"omega_{motor_idx}_rpm"
        if col in d:
            axes[0].plot(d["time_s"], d[col], lw=0.95, label=f"M{motor_idx}", color=color)
    axes[0].set_title("电机转速时序")
    axes[0].set_ylabel("omega / rpm")
    axes[0].legend(ncol=min(len(motors), 4), fontsize=9)

    if "omega_mean_rpm" in d:
        axes[1].plot(d["time_s"], d["omega_mean_rpm"], label="平均转速", lw=1.2, color="#111111")
    if "omega_spread_rpm" in d:
        axes[1].plot(d["time_s"], d["omega_spread_rpm"], label="转速极差", lw=1.0, color="#d62728")
    axes[1].set_title("平均转速与电机间离散度")
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("rpm")
    axes[1].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    path = out / "08_motor_omega.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_motor_cmd_vs_omega(df: pd.DataFrame, out: Path, max_points: int) -> Path | None:
    motors = active_motor_indices_from_frame(df)
    motors = [idx for idx in motors if f"motor_cmd_{idx}" in df and f"omega_{idx}_rpm" in df]
    if not motors:
        return None
    idx = plot_indices(len(df), max_points)
    d = df.iloc[idx]
    ncols = 2
    nrows = int(math.ceil(len(motors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, max(4.5, 4.2 * nrows)), squeeze=False)
    axes_flat = axes.ravel()
    for ax, motor_idx in zip(axes_flat, motors):
        cmd_col = f"motor_cmd_{motor_idx}"
        rpm_col = f"omega_{motor_idx}_rpm"
        sc = ax.scatter(d[cmd_col], d[rpm_col], c=d["time_s"], s=8, cmap="viridis", alpha=0.6)
        ax.set_xlabel("电机命令")
        ax.set_ylabel("omega / rpm")
        ax.set_title(f"M{motor_idx}: 命令-转速, r={corr(d[cmd_col], d[rpm_col]):.3f}")
        fig.colorbar(sc, ax=ax, label="时间 (s)")
    for ax in axes_flat[len(motors):]:
        ax.axis("off")
    fig.tight_layout()
    path = out / "09_motor_cmd_vs_omega.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_motor_balance(df: pd.DataFrame, out: Path, max_points: int) -> Path | None:
    motors = active_motor_indices_from_frame(df)
    omega_cols = [f"omega_{idx}_rpm" for idx in motors if f"omega_{idx}_rpm" in df]
    if not omega_cols:
        return None
    idx = plot_indices(len(df), max_points)
    d = df.iloc[idx]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].boxplot(
        [finite(df[col]) for col in omega_cols],
        tick_labels=[col.replace("omega_", "M").replace("_rpm", "") for col in omega_cols],
        showfliers=False,
    )
    axes[0].set_title("各电机转速分布")
    axes[0].set_xlabel("电机")
    axes[0].set_ylabel("omega / rpm")

    if "omega_spread_rpm" in d:
        axes[1].plot(d["time_s"], d["omega_spread_rpm"], lw=1.0, color="#d62728", label="转速极差")
        p95 = float(np.nanpercentile(df["omega_spread_rpm"].to_numpy(dtype=float), 95))
        axes[1].axhline(p95, lw=0.9, ls="--", color="#111111", label=f"P95={p95:.1f} rpm")
    axes[1].set_title("电机间转速离散度")
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("rpm")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    path = out / "10_motor_balance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_plots(
    df: pd.DataFrame,
    paths: AnalysisPaths,
    max_points: int,
    gate_passages: pd.DataFrame,
) -> list[Path]:
    figures = [
        plot_speed_height(df, paths.figures_dir, max_points),
        plot_trajectory(df, paths.figures_dir, max_points),
        plot_trajectory_heading_overview(df, paths.figures_dir, max_points),
        plot_yaw(df, paths.figures_dir, max_points),
    ]
    err_fig = plot_errors(df, paths.figures_dir, max_points)
    if err_fig:
        figures.append(err_fig)
    figures.append(
        plot_trajectory_gate_3d(
            [("SITL", "#0072B2", df)],
            [gate_passages],
            GATES,
            GATE_LABELS,
            paths.figures_dir / "05_trajectory_3d_with_gates.png",
            max_points,
            "SITL 三维轨迹与 1.5 m × 1.5 m 门框",
        )
    )
    for motor_figure in (
        plot_motor_commands(df, paths.figures_dir, max_points),
        plot_motor_omega(df, paths.figures_dir, max_points),
        plot_motor_cmd_vs_omega(df, paths.figures_dir, max_points),
        plot_motor_balance(df, paths.figures_dir, max_points),
    ):
        if motor_figure:
            figures.append(motor_figure)
    return figures


def make_paths(log_path: Path, out_root: Path, label: str | None) -> AnalysisPaths:
    safe_name = label or log_path.stem
    safe_name = safe_name.replace(" ", "_").replace(".", "_")
    out_dir = out_root / f"{safe_name}_sitl_analysis"
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return AnalysisPaths(
        out_dir=out_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        summary_json=out_dir / "summary.json",
    )


def export_outputs(df: pd.DataFrame, metrics: pd.DataFrame, summary: dict, paths: AnalysisPaths) -> None:
    df.to_csv(paths.tables_dir / "normalized_timeseries.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(paths.tables_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    with paths.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze SITL CSV logs: speed, height, trajectory, yaw, and figures.")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="SITL CSV path.")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output root directory.")
    ap.add_argument("--label", default=None, help="Optional output folder/report label.")
    ap.add_argument("--track-config", default=None, help="JSON track with start and gates as [x, y, z(NED), yaw(deg)].")
    ap.add_argument("--position-source", choices=("pos", "extPos", "ekf_pos"), default=None, help="Position fields; default priority: extPos, pos, ekf_pos.")
    ap.add_argument("--velocity-source", choices=("vel", "extVel", "ekf_vel"), default=None, help="Velocity fields; default priority: extVel, vel, ekf_vel.")
    ap.add_argument("--start-s", type=float, default=None, help="Optional start time in seconds after log start.")
    ap.add_argument("--end-s", type=float, default=None, help="Optional end time in seconds after log start.")
    ap.add_argument("--max-plot-points", type=int, default=MAX_PLOT_POINTS, help="Downsample plots to this many points.")
    return ap.parse_args()


def main() -> int:
    configure_matplotlib()
    args = parse_args()
    track = None
    if args.track_config:
        track_path = Path(args.track_config)
        if not track_path.is_absolute():
            track_path = ROOT / track_path
        track = json.loads(track_path.read_text(encoding="utf-8"))
        gates = np.asarray(track["gates"], dtype=float)
        if gates.ndim != 2 or gates.shape[1] != 4 or len(gates) == 0 or not np.isfinite(gates).all():
            raise ValueError("Track gates must be a nonempty finite N x 4 array: [x, y, z(NED), yaw(deg)].")
        global GATES, GATE_LABELS
        GATES = gates.copy()
        GATES[:, 3] = np.deg2rad(GATES[:, 3])
        GATE_LABELS = [f"Gate {idx:02d}" for idx in range(1, len(GATES) + 1)]
    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    log_path = log_path.resolve()
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    paths = make_paths(log_path, out_root.resolve(), args.label)

    df, profile = load_sitl_frame(log_path, start_s=args.start_s, end_s=args.end_s,
                                position_source=args.position_source, velocity_source=args.velocity_source)
    metrics, summary = make_metrics(df, profile)
    gate_passages = evaluate_gate_passages(df, GATES, GATE_LABELS, args.label or log_path.stem)
    summary["gates_passed_3d"] = int(gate_passages["passed"].sum())
    summary["gates_total_3d"] = int(len(gate_passages))
    if track is not None:
        summary["track_config"] = track
        summary["track_yaw_unit"] = "deg"
    export_outputs(df, metrics, summary, paths)
    gate_passages.to_csv(paths.tables_dir / "gate_passage_3d.csv", index=False, encoding="utf-8-sig")
    figures = generate_plots(
        df,
        paths,
        max_points=max(100, args.max_plot_points),
        gate_passages=gate_passages,
    )

    print(f"Output directory: {paths.out_dir}")
    print(f"State sources: position={profile['position_prefix']}, velocity={profile['velocity_prefix']}")
    print(f"Figures: {paths.figures_dir}")
    print("Generated figures:")
    for figure in figures:
        print(f"  {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
