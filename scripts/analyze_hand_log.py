from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "data" / "raw" / "hand" / "hand_0629_lyh.csv"
DEFAULT_OUT_ROOT = ROOT / "reports"
MAX_PLOT_POINTS = 5000


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


def vector_cols(columns: set[str], prefix: str, n: int = 3) -> list[str]:
    out = [f"{prefix}[{idx}]" for idx in range(n)]
    return out if all(col in columns for col in out) else []


def first_vector(columns: set[str], prefixes: Iterable[str], n: int = 3) -> tuple[str | None, list[str]]:
    for prefix in prefixes:
        cols = vector_cols(columns, prefix, n=n)
        if cols:
            return prefix, cols
    return None, []


def wanted_columns(columns: list[str]) -> list[str]:
    available = set(columns)
    wanted = [
        "timeS",
        "timeUs",
        "baroAlt",
        "flightModeFlags",
        "stateFlags",
        "failsafePhase",
        "rxSignalReceived",
        "rxFlightChannelsValid",
    ]
    for prefix in (
        "pos",
        "extPos",
        "ekf_pos",
        "vel",
        "extVel",
        "ekf_vel",
        "quat",
        "extAtt",
        "ekf_att",
        "gyroADC",
        "gyroSp",
        "motor",
        "omegaUnfiltered",
        "rcCommand",
        "setpoint",
    ):
        n = 4 if prefix in {"quat", "motor", "omegaUnfiltered", "rcCommand", "setpoint"} else 3
        wanted.extend(f"{prefix}[{idx}]" for idx in range(n))
    return [col for col in wanted if col in available]


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


def plot_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def load_hand_frame(path: Path, start_s: float | None = None, end_s: float | None = None) -> tuple[pd.DataFrame, dict]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    raw = pd.read_csv(path, usecols=wanted_columns(columns))
    available = set(raw.columns)

    if "timeS" in raw:
        time_s = pd.to_numeric(raw["timeS"], errors="coerce").astype(float)
        time_s = time_s - float(time_s.iloc[0])
    elif "timeUs" in raw:
        time_s = pd.to_numeric(raw["timeUs"], errors="coerce").astype(float)
        time_s = (time_s - float(time_s.iloc[0])) / 1_000_000.0
    else:
        raise ValueError("CSV needs timeS or timeUs.")

    pos_prefix, pos_cols = first_vector(available, ("extPos", "ekf_pos", "pos"))
    vel_prefix, vel_cols = first_vector(available, ("extVel", "ekf_vel", "vel"))
    _, quat_cols = first_vector(available, ("quat",), n=4)
    att_prefix, att_cols = first_vector(available, ("extAtt", "ekf_att"), n=3)
    if not pos_cols:
        raise ValueError("CSV needs extPos, ekf_pos, or pos columns.")
    if not vel_cols:
        raise ValueError("CSV needs extVel, ekf_vel, or vel columns.")
    if not quat_cols and not att_cols:
        raise ValueError("CSV needs quat or attitude columns.")

    df = pd.DataFrame({"time_s": time_s})
    for axis, col in zip(("x", "y", "z"), pos_cols):
        df[f"pos_{axis}_m"] = pd.to_numeric(raw[col], errors="coerce")
    for axis, col in zip(("x", "y", "z"), vel_cols):
        df[f"vel_{axis}_mps"] = pd.to_numeric(raw[col], errors="coerce")

    if quat_cols:
        euler = quat_to_euler_rad(raw[quat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
        df["roll_rad"] = euler[:, 0]
        df["pitch_rad"] = euler[:, 1]
        df["yaw_rad"] = wrap_pi(euler[:, 2])
        attitude_source = quat_cols
    else:
        att = raw[att_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.nanmedian(np.abs(att)) > np.pi * 1.5:
            att = np.deg2rad(att)
        df["roll_rad"] = att[:, 0]
        df["pitch_rad"] = att[:, 1]
        df["yaw_rad"] = wrap_pi(att[:, 2])
        attitude_source = att_cols

    z_sign = -1.0 if float(np.nanmedian(df["pos_z_m"])) < 0.0 else 1.0
    df["height_m"] = z_sign * df["pos_z_m"]
    df["speed_xy_mps"] = np.hypot(df["vel_x_mps"], df["vel_y_mps"])
    df["speed_3d_mps"] = np.sqrt(df["vel_x_mps"] ** 2 + df["vel_y_mps"] ** 2 + df["vel_z_mps"] ** 2)
    df["roll_deg"] = np.rad2deg(df["roll_rad"])
    df["pitch_deg"] = np.rad2deg(df["pitch_rad"])
    df["yaw_deg"] = np.rad2deg(df["yaw_rad"])
    df["yaw_unwrapped_deg"] = np.rad2deg(np.unwrap(df["yaw_rad"].to_numpy(dtype=float)))

    for i, axis in enumerate(("x", "y", "z")):
        col = f"gyroADC[{i}]"
        if col in raw:
            gyro = pd.to_numeric(raw[col], errors="coerce")
            df[f"gyro_{axis}_dps"] = np.rad2deg(gyro) if float(np.nanmedian(np.abs(gyro))) < 20.0 else gyro
        sp_col = f"gyroSp[{i}]"
        if sp_col in raw:
            gyro_sp = pd.to_numeric(raw[sp_col], errors="coerce")
            df[f"gyro_sp_{axis}_dps"] = np.rad2deg(gyro_sp) if float(np.nanmedian(np.abs(gyro_sp))) < 20.0 else gyro_sp

    for i in range(4):
        m_col = f"motor[{i}]"
        o_col = f"omegaUnfiltered[{i}]"
        if m_col in raw:
            df[f"motor_{i}"] = pd.to_numeric(raw[m_col], errors="coerce")
        if o_col in raw:
            df[f"omega_{i}_rpm"] = pd.to_numeric(raw[o_col], errors="coerce")

    for col in ("baroAlt", "flightModeFlags", "stateFlags", "rxSignalReceived", "rxFlightChannelsValid"):
        if col in raw:
            df[col] = pd.to_numeric(raw[col], errors="coerce")

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
        "attitude_source": attitude_source,
        "height_convention": "NED z down, height = -z" if z_sign < 0 else "z up, height = z",
        "position_prefix": pos_prefix,
        "velocity_prefix": vel_prefix,
        "attitude_prefix": att_prefix,
        "time_window_s": [start_s, end_s],
    }
    return df, profile


def make_metrics(df: pd.DataFrame, profile: dict) -> tuple[pd.DataFrame, dict]:
    dpos = np.diff(df[["pos_x_m", "pos_y_m", "pos_z_m"]].to_numpy(dtype=float), axis=0)
    path_len_xy = float(np.nansum(np.linalg.norm(dpos[:, :2], axis=1))) if len(dpos) else float("nan")
    path_len_3d = float(np.nansum(np.linalg.norm(dpos, axis=1))) if len(dpos) else float("nan")
    rows: list[dict[str, str | float]] = []

    def add(group: str, signal: str, unit: str, values: pd.Series | np.ndarray) -> None:
        row = {"group": group, "signal": signal, "unit": unit}
        row.update(stats(values))
        rows.append(row)

    for signal, unit, col in (
        ("horizontal speed", "m/s", "speed_xy_mps"),
        ("3D speed", "m/s", "speed_3d_mps"),
        ("height", "m", "height_m"),
        ("roll", "deg", "roll_deg"),
        ("pitch", "deg", "pitch_deg"),
        ("yaw", "deg", "yaw_deg"),
        ("yaw unwrapped", "deg", "yaw_unwrapped_deg"),
    ):
        add("motion", signal, unit, df[col])
    for axis in ("x", "y", "z"):
        col = f"gyro_{axis}_dps"
        if col in df:
            add("gyro", f"gyro {axis}", "deg/s", df[col])
    for i in range(4):
        if f"motor_{i}" in df:
            add("motor", f"motor {i} command", "norm", df[f"motor_{i}"])
        if f"omega_{i}_rpm" in df:
            add("motor", f"motor {i} omega", "rpm", df[f"omega_{i}_rpm"])

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
        "roll_abs_p95_deg": float(np.percentile(np.abs(finite(df["roll_deg"])), 95)),
        "pitch_abs_p95_deg": float(np.percentile(np.abs(finite(df["pitch_deg"])), 95)),
        "yaw_total_rotation_deg": float(df["yaw_unwrapped_deg"].iloc[-1] - df["yaw_unwrapped_deg"].iloc[0]),
    }
    return pd.DataFrame(rows), summary


def make_paths(log_path: Path, out_root: Path, label: str | None) -> AnalysisPaths:
    safe_name = (label or log_path.stem).replace(" ", "_").replace(".", "_")
    out_dir = out_root / f"{safe_name}_hand_analysis"
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return AnalysisPaths(out_dir, figures_dir, tables_dir, out_dir / "summary.json")


def plot_trajectory_heading(df: pd.DataFrame, out: Path, max_points: int) -> Path:
    d = df.iloc[plot_indices(len(df), max_points)]
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], wspace=0.18, hspace=0.32)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_yaw = fig.add_subplot(gs[1, 1])
    color = "#1f77b4"
    ax_xy.plot(d["pos_x_m"], d["pos_y_m"], color=color, lw=1.2, label="Hand")
    q = d.iloc[:: max(1, len(d) // 18)]
    ax_xy.quiver(
        q["pos_x_m"],
        q["pos_y_m"],
        np.cos(q["yaw_rad"]),
        np.sin(q["yaw_rad"]),
        color=color,
        alpha=0.32,
        width=0.0035,
        scale=16,
        headwidth=4.0,
        headlength=5.0,
        label="机头朝向",
    )
    ax_xy.scatter(d["pos_x_m"].iloc[0], d["pos_y_m"].iloc[0], marker="o", s=55, color="#2ca02c", label="Start")
    ax_xy.scatter(d["pos_x_m"].iloc[-1], d["pos_y_m"].iloc[-1], marker="X", s=70, color="#d62728", label="End")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("X 位置 (m)")
    ax_xy.set_ylabel("Y 位置 (m)")
    ax_xy.set_title("XY 手持轨迹")
    ax_xy.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)

    ax_h.plot(d["time_s"], d["height_m"], color=color, lw=1.1, label="高度")
    if "baroAlt" in d:
        ax_h_t = ax_h.twinx()
        ax_h_t.plot(d["time_s"], d["baroAlt"], color="#ff7f0e", lw=0.8, alpha=0.65, label="baroAlt")
        ax_h_t.set_ylabel("baroAlt")
    ax_h.set_title("高度")
    ax_h.set_xlabel("时间 (s)")
    ax_h.set_ylabel("高度 (m)")
    ax_h.legend(fontsize=8, loc="upper right")

    ax_yaw.plot(d["time_s"], d["yaw_deg"], color=color, lw=1.0, label="航向角")
    ax_yaw.set_title("航向角时序")
    ax_yaw.set_xlabel("时间 (s)")
    ax_yaw.set_ylabel("航向角 (deg)")
    ax_yaw.legend(fontsize=8, loc="upper right")

    fig.suptitle("手持无桨实验：轨迹、高度与航向", fontsize=15, fontweight="bold")
    path = out / "01_hand_trajectory_heading.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_position_velocity(df: pd.DataFrame, out: Path, max_points: int) -> Path:
    d = df.iloc[plot_indices(len(df), max_points)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for axis in ("x", "y"):
        axes[0].plot(d["time_s"], d[f"pos_{axis}_m"], lw=1.0, label=axis.upper())
    axes[0].plot(d["time_s"], d["height_m"], lw=1.0, label="Height")
    axes[0].set_ylabel("Position / height (m)")
    axes[0].set_title("Position components")
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].plot(d["time_s"], d["speed_xy_mps"], lw=1.1, label="Horizontal speed")
    axes[1].plot(d["time_s"], d["speed_3d_mps"], lw=1.0, label="3D speed")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Handheld speed")
    axes[1].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    path = out / "02_position_velocity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_attitude_gyro(df: pd.DataFrame, out: Path, max_points: int) -> Path:
    d = df.iloc[plot_indices(len(df), max_points)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for col, label in (("roll_deg", "Roll"), ("pitch_deg", "Pitch"), ("yaw_deg", "Yaw")):
        axes[0].plot(d["time_s"], d[col], lw=0.95, label=label)
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Attitude")
    axes[0].legend(ncol=3, fontsize=9)
    for axis in ("x", "y", "z"):
        col = f"gyro_{axis}_dps"
        if col in d:
            axes[1].plot(d["time_s"], d[col], lw=0.9, label=f"Gyro {axis}")
    axes[1].set_ylabel("Angular rate (deg/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Gyroscope")
    axes[1].legend(ncol=3, fontsize=9)
    fig.tight_layout()
    path = out / "03_attitude_gyro.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_motor_omega(df: pd.DataFrame, out: Path, max_points: int) -> Path | None:
    motor_cols = [f"motor_{i}" for i in range(4) if f"motor_{i}" in df]
    omega_cols = [f"omega_{i}_rpm" for i in range(4) if f"omega_{i}_rpm" in df]
    if not motor_cols and not omega_cols:
        return None
    d = df.iloc[plot_indices(len(df), max_points)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for col in motor_cols:
        axes[0].plot(d["time_s"], d[col], lw=0.9, label=col)
    axes[0].set_ylabel("Motor command")
    axes[0].set_title("Motor command channels")
    axes[0].legend(ncol=4, fontsize=8)
    for col in omega_cols:
        axes[1].plot(d["time_s"], d[col], lw=0.85, label=col)
    axes[1].set_ylabel("RPM")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Motor omega unfiltered")
    axes[1].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    path = out / "04_motor_omega.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_trajectory_3d(df: pd.DataFrame, out: Path, max_points: int) -> Path:
    d = df.iloc[plot_indices(len(df), max_points)]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(d["pos_x_m"], d["pos_y_m"], d["height_m"], lw=1.0, color="#1f77b4")
    ax.scatter(d["pos_x_m"].iloc[0], d["pos_y_m"].iloc[0], d["height_m"].iloc[0], s=60, color="#2ca02c", label="Start")
    ax.scatter(d["pos_x_m"].iloc[-1], d["pos_y_m"].iloc[-1], d["height_m"].iloc[-1], s=70, marker="X", color="#d62728", label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title("3D handheld trajectory")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out / "05_trajectory_3d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def export_outputs(df: pd.DataFrame, metrics: pd.DataFrame, summary: dict, paths: AnalysisPaths) -> None:
    df.to_csv(paths.tables_dir / "normalized_timeseries.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(paths.tables_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    with paths.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze handheld no-prop CSV logs and generate figures.")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="Handheld CSV path.")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output root directory.")
    ap.add_argument("--label", default=None, help="Optional output folder/report label.")
    ap.add_argument("--start-s", type=float, default=None, help="Optional start time after log start.")
    ap.add_argument("--end-s", type=float, default=None, help="Optional end time after log start.")
    ap.add_argument("--max-plot-points", type=int, default=MAX_PLOT_POINTS, help="Downsample plots to this many points.")
    return ap.parse_args()


def main() -> int:
    configure_matplotlib()
    args = parse_args()
    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    paths = make_paths(log_path.resolve(), out_root.resolve(), args.label)

    df, profile = load_hand_frame(log_path.resolve(), start_s=args.start_s, end_s=args.end_s)
    metrics, summary = make_metrics(df, profile)
    export_outputs(df, metrics, summary, paths)
    figures = [
        plot_trajectory_heading(df, paths.figures_dir, max(100, args.max_plot_points)),
        plot_position_velocity(df, paths.figures_dir, max(100, args.max_plot_points)),
        plot_attitude_gyro(df, paths.figures_dir, max(100, args.max_plot_points)),
        plot_trajectory_3d(df, paths.figures_dir, max(100, args.max_plot_points)),
    ]
    motor_fig = plot_motor_omega(df, paths.figures_dir, max(100, args.max_plot_points))
    if motor_fig:
        figures.insert(3, motor_fig)

    print(f"Output directory: {paths.out_dir}")
    print(f"Figures: {paths.figures_dir}")
    print("Generated figures:")
    for figure in figures:
        print(f"  {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
