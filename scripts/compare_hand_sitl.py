from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_hand_log import load_hand_frame  # noqa: E402
from analyze_sitl_log import load_sitl_frame  # noqa: E402
from track_3d import (  # noqa: E402
    DEFAULT_GATE_LABELS,
    DEFAULT_GATES_NED,
    evaluate_gate_passages,
    normalize_gate_labels,
    normalize_gates,
    plot_trajectory_gate_3d,
)


DEFAULT_HAND = ROOT / "data" / "raw" / "hand" / "hand_0629_lyh.csv"
DEFAULT_SITL = ROOT / "data" / "raw" / "sitl" / "LOG_SITL_0628lyh.csv"
DEFAULT_OUT_ROOT = ROOT / "reports"
DEFAULT_CONFIG = ROOT / "configs" / "hand_sitl_compare_config.json"
MAX_PLOT_POINTS = 6000
MOTOR_SATURATION = 0.95


@dataclass
class Dataset:
    name: str
    color: str
    df: pd.DataFrame
    profile: dict


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cfg_value(config: dict[str, Any], key: str, cli_value: Any, default: Any) -> Any:
    return cli_value if cli_value is not None else config.get(key, default)


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select_active_segment(
    df: pd.DataFrame,
    profile: dict,
    domain: str,
    hand_flags: tuple[int, ...],
    speed_threshold: float,
    displacement_threshold: float,
) -> tuple[pd.DataFrame, dict, str]:
    if domain == "hand" and "flightModeFlags" in df and hand_flags:
        mask = df["flightModeFlags"].fillna(-1).astype(int).isin(hand_flags).to_numpy()
        if mask.any():
            note = f"手持按 flightModeFlags in {list(hand_flags)} 截取控制段"
        else:
            mask = np.zeros(len(df), dtype=bool)
            note = ""
    else:
        mask = np.zeros(len(df), dtype=bool)
        note = ""

    if not mask.any():
        speed = df["speed_xy_mps"].to_numpy(dtype=float)
        pos = df[["pos_x_m", "pos_y_m"]].to_numpy(dtype=float)
        displacement = np.linalg.norm(pos - pos[0], axis=1)
        candidates = (speed > speed_threshold) | (displacement > displacement_threshold)
        indices = np.flatnonzero(candidates)
        if len(indices):
            mask[indices[0] : indices[-1] + 1] = True
            note = (
                f"{domain} 按速度 > {speed_threshold:.2f} m/s 或偏离起点 "
                f"> {displacement_threshold:.2f} m 自动截取连续运动段"
            )
        else:
            mask[:] = True
            note = f"{domain} 未检测到明显运动，使用全段"

    selected = df.loc[mask].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"{domain} selected segment is empty")
    start_s = float(selected["time_s"].iloc[0])
    end_s = float(selected["time_s"].iloc[-1])
    selected["time_s"] = selected["time_s"] - start_s
    updated = dict(profile)
    updated.update(
        {
            "rows_selected": int(len(selected)),
            "selected_start_s": start_s,
            "selected_end_s": end_s,
            "selected_duration_s": end_s - start_s,
            "segment_note": note,
        }
    )
    return selected, updated, note


def add_aggregate_columns(df: pd.DataFrame) -> None:
    for axis in ("roll", "pitch", "yaw"):
        radians = f"{axis}_rad"
        degrees = f"{axis}_deg"
        if degrees not in df and radians in df:
            df[degrees] = np.rad2deg(df[radians])
    motor_cols = [f"motor_{i}" for i in range(4) if f"motor_{i}" in df]
    omega_cols = [f"omega_{i}_rpm" for i in range(4) if f"omega_{i}_rpm" in df]
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


def configure_matplotlib() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.28


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def stats(values: pd.Series | np.ndarray) -> dict[str, float]:
    arr = finite(values)
    if len(arr) == 0:
        return {k: float("nan") for k in ("mean", "p50", "p95", "min", "max")}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def plot_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    return df.iloc[plot_indices(len(df), max_points)].copy()


def add_motor_omega_columns(df: pd.DataFrame, path: Path) -> None:
    header = list(pd.read_csv(path, nrows=0).columns)
    usecols = [col for col in [*(f"motor[{i}]" for i in range(4)), *(f"omegaUnfiltered[{i}]" for i in range(4))] if col in header]
    if not usecols:
        return
    raw = pd.read_csv(path, usecols=usecols)
    if len(raw) != len(df):
        raise ValueError(f"Motor/RPM row count mismatch for {path}: {len(raw)} raw rows vs {len(df)} analysis rows")
    for i in range(4):
        motor_col = f"motor[{i}]"
        omega_col = f"omegaUnfiltered[{i}]"
        if motor_col in raw:
            df[f"motor_{i}"] = pd.to_numeric(raw[motor_col], errors="coerce")
        if omega_col in raw:
            df[f"omega_{i}_rpm"] = pd.to_numeric(raw[omega_col], errors="coerce")
    omega_cols = [f"omega_{i}_rpm" for i in range(4) if f"omega_{i}_rpm" in df]
    motor_cols = [f"motor_{i}" for i in range(4) if f"motor_{i}" in df]
    if omega_cols:
        df["omega_mean_rpm"] = df[omega_cols].mean(axis=1)
        df["omega_spread_rpm"] = df[omega_cols].max(axis=1) - df[omega_cols].min(axis=1)
    if motor_cols:
        df["motor_mean"] = df[motor_cols].mean(axis=1)


def norm_time(df: pd.DataFrame) -> pd.Series:
    duration = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0])
    if duration <= 0:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return (df["time_s"] - float(df["time_s"].iloc[0])) / duration


def common_xy_limits(datasets: list[Dataset], margin_ratio: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = np.concatenate([finite(ds.df["pos_x_m"]) for ds in datasets])
    ys = np.concatenate([finite(ds.df["pos_y_m"]) for ds in datasets])
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    span = span * (1.0 + margin_ratio) if span > 0 else 1.0
    half = span / 2.0
    return (cx - half, cx + half), (cy - half, cy + half)


def common_y_limits(datasets: list[Dataset], col: str, margin_ratio: float = 0.08) -> tuple[float, float]:
    vals = np.concatenate([finite(ds.df[col]) for ds in datasets])
    ymin, ymax = float(np.min(vals)), float(np.max(vals))
    span = ymax - ymin
    if span <= 0:
        span = 1.0
    margin = span * margin_ratio
    return ymin - margin, ymax + margin


def apply_common_xy(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))


def arrow_quiver(ax, df: pd.DataFrame, color: str, max_arrows: int = 18) -> None:
    step = max(1, len(df) // max_arrows)
    q = df.iloc[::step]
    ax.quiver(
        q["pos_x_m"],
        q["pos_y_m"],
        np.cos(q["yaw_rad"]),
        np.sin(q["yaw_rad"]),
        color=color,
        alpha=0.24,
        width=0.0032,
        scale=16,
        headwidth=4.0,
        headlength=5.0,
    )


def make_summary(datasets: list[Dataset]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for ds in datasets:
        df = ds.df
        dpos = np.diff(df[["pos_x_m", "pos_y_m", "pos_z_m"]].to_numpy(dtype=float), axis=0)
        path_xy = float(np.nansum(np.linalg.norm(dpos[:, :2], axis=1))) if len(dpos) else float("nan")
        path_3d = float(np.nansum(np.linalg.norm(dpos, axis=1))) if len(dpos) else float("nan")
        speed = stats(df["speed_xy_mps"])
        height = stats(df["height_m"])
        omega = stats(df["omega_mean_rpm"]) if "omega_mean_rpm" in df else {}
        spread = stats(df["omega_spread_rpm"]) if "omega_spread_rpm" in df else {}
        motor_max = stats(df["motor_max"]) if "motor_max" in df else {}
        motor_spread = stats(df["motor_spread"]) if "motor_spread" in df else {}
        rows.append(
            {
                "dataset": ds.name,
                "duration_s": float(df["time_s"].iloc[-1]),
                "rows": int(len(df)),
                "approx_hz": float(ds.profile.get("approx_hz", np.nan)),
                "path_xy_m": path_xy,
                "path_3d_m": path_3d,
                "x_range_m": float(df["pos_x_m"].max() - df["pos_x_m"].min()),
                "y_range_m": float(df["pos_y_m"].max() - df["pos_y_m"].min()),
                "height_min_m": height["min"],
                "height_max_m": height["max"],
                "height_mean_m": height["mean"],
                "speed_mean_mps": speed["mean"],
                "speed_p95_mps": speed["p95"],
                "speed_max_mps": speed["max"],
                "roll_abs_p95_deg": float(np.nanpercentile(np.abs(df["roll_deg"]), 95)),
                "pitch_abs_p95_deg": float(np.nanpercentile(np.abs(df["pitch_deg"]), 95)),
                "omega_mean_rpm": omega.get("mean", float("nan")),
                "omega_p50_rpm": omega.get("p50", float("nan")),
                "omega_p95_rpm": omega.get("p95", float("nan")),
                "omega_max_rpm": omega.get("max", float("nan")),
                "omega_spread_p95_rpm": spread.get("p95", float("nan")),
                "motor_cmd_max_p95": motor_max.get("p95", float("nan")),
                "motor_cmd_max_max": motor_max.get("max", float("nan")),
                "motor_spread_p95": motor_spread.get("p95", float("nan")),
                "motor_any_high_sat_rate": float((df["motor_max"] >= MOTOR_SATURATION).mean()) if "motor_max" in df else float("nan"),
                "yaw_total_rotation_deg": float(df["yaw_unwrapped_deg"].iloc[-1] - df["yaw_unwrapped_deg"].iloc[0]),
                "position_source": ds.profile.get("position_prefix", ""),
                "velocity_source": ds.profile.get("velocity_prefix", ""),
            }
        )
    return pd.DataFrame(rows)


def plot_xy_side_by_side(datasets: list[Dataset], out: Path, max_points: int) -> Path:
    xlim, ylim = common_xy_limits(datasets)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
    for ax, ds in zip(axes, datasets):
        d = downsample(ds.df, max_points)
        ax.plot(d["pos_x_m"], d["pos_y_m"], color=ds.color, lw=1.15, label=ds.name)
        arrow_quiver(ax, d, ds.color)
        ax.scatter(d["pos_x_m"].iloc[0], d["pos_y_m"].iloc[0], s=48, color="#2ca02c", label="Start")
        ax.scatter(d["pos_x_m"].iloc[-1], d["pos_y_m"].iloc[-1], s=60, marker="X", color="#d62728", label="End")
        apply_common_xy(ax, xlim, ylim)
        ax.set_title(f"{ds.name} XY轨迹")
        ax.set_xlabel("X 位置 (m)")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("Y 位置 (m)")
    fig.suptitle("XY轨迹对比：左右子图同坐标范围/同刻度", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = out / "01_xy_side_by_side_same_scale.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_xy_overlay(datasets: list[Dataset], out: Path, max_points: int) -> Path:
    xlim, ylim = common_xy_limits(datasets)
    fig, ax = plt.subplots(figsize=(8, 7))
    for ds in datasets:
        d = downsample(ds.df, max_points)
        ax.plot(d["pos_x_m"], d["pos_y_m"], color=ds.color, lw=1.2, label=ds.name)
        arrow_quiver(ax, d, ds.color, max_arrows=14)
    apply_common_xy(ax, xlim, ylim)
    ax.set_xlabel("X 位置 (m)")
    ax.set_ylabel("Y 位置 (m)")
    ax.set_title("XY轨迹叠加：同一坐标系")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out / "02_xy_overlay_same_axis.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_series_side_by_side(
    datasets: list[Dataset],
    out: Path,
    col: str,
    ylabel: str,
    title: str,
    filename: str,
    extra_cols: tuple[str, str] | None = None,
    max_points: int = MAX_PLOT_POINTS,
) -> Path:
    ylim = common_y_limits(datasets, col)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    for ax, ds in zip(axes, datasets):
        d = downsample(ds.df, max_points)
        ax.plot(norm_time(d), d[col], color=ds.color, lw=1.1, label=ds.name)
        if extra_cols and extra_cols[0] in d:
            ax.plot(norm_time(d), d[extra_cols[0]], color="#555555", lw=0.9, ls="--", label=extra_cols[1])
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.set_title(ds.name)
        ax.set_xlabel("归一化时间")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{title}：左右子图同纵轴范围/同刻度", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_series_overlay(
    datasets: list[Dataset],
    out: Path,
    col: str,
    ylabel: str,
    title: str,
    filename: str,
    max_points: int = MAX_PLOT_POINTS,
) -> Path:
    ylim = common_y_limits(datasets, col)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for ds in datasets:
        d = downsample(ds.df, max_points)
        ax.plot(norm_time(d), d[col], color=ds.color, lw=1.1, label=ds.name)
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_xlabel("归一化时间")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}叠加：同一坐标轴")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_omega_channels_side_by_side(datasets: list[Dataset], out: Path, max_points: int) -> Path:
    omega_cols = [f"omega_{i}_rpm" for i in range(4)]
    vals = []
    for ds in datasets:
        for col in omega_cols:
            if col in ds.df:
                vals.append(finite(ds.df[col]))
    all_vals = np.concatenate([v for v in vals if len(v)]) if vals else np.array([0.0, 1.0])
    ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
    span = max(ymax - ymin, 1.0)
    ylim = (ymin - 0.06 * span, ymax + 0.06 * span)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for ax, ds in zip(axes, datasets):
        d = downsample(ds.df, max_points)
        for i, col in enumerate(omega_cols):
            if col in d:
                ax.plot(norm_time(d), d[col], lw=0.85, label=f"M{i}")
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.set_title(ds.name)
        ax.set_xlabel("归一化时间")
        ax.legend(fontsize=8, ncol=4, loc="upper right")
    axes[0].set_ylabel("电机转速 (rpm)")
    fig.suptitle("四电机转速通道对比：左右子图同纵轴范围/同刻度", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = out / "13_omega_channels_side_by_side_same_scale.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_omega_channels_overlay(datasets: list[Dataset], out: Path, max_points: int) -> Path:
    omega_cols = [f"omega_{i}_rpm" for i in range(4)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    vals = []
    for ds in datasets:
        for col in omega_cols:
            if col in ds.df:
                vals.append(finite(ds.df[col]))
    all_vals = np.concatenate([v for v in vals if len(v)]) if vals else np.array([0.0, 1.0])
    ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
    span = max(ymax - ymin, 1.0)
    ylim = (ymin - 0.06 * span, ymax + 0.06 * span)

    for i, ax in enumerate(axes):
        col = f"omega_{i}_rpm"
        for ds in datasets:
            if col in ds.df:
                d = downsample(ds.df, max_points)
                ax.plot(norm_time(d), d[col], color=ds.color, lw=0.9, label=ds.name)
        ax.set_title(f"Motor {i}")
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.legend(fontsize=8)
    for ax in axes[2:]:
        ax.set_xlabel("归一化时间")
    for ax in axes[::2]:
        ax.set_ylabel("rpm")
    fig.suptitle("四电机转速叠加：同一坐标轴/同刻度", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = out / "14_omega_channels_overlay_same_axis.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_gate_metrics(datasets: list[Dataset], gates: np.ndarray, labels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    if len(gates) == 0:
        return pd.DataFrame()
    normalized_labels = list(labels[: len(gates)])
    normalized_labels.extend(f"Gate {i + 1}" for i in range(len(normalized_labels), len(gates)))
    for ds in datasets:
        xy = ds.df[["pos_x_m", "pos_y_m"]].to_numpy(dtype=float)
        for label, gate in zip(normalized_labels, gates):
            gx, gy = float(gate[0]), float(gate[1])
            distance = np.hypot(xy[:, 0] - gx, xy[:, 1] - gy)
            closest = int(np.nanargmin(distance))
            inside = distance < 0.55
            entries = int(np.sum(inside & ~np.r_[False, inside[:-1]]))
            rows.append(
                {
                    "dataset": ds.name,
                    "gate": label,
                    "gate_x_m": gx,
                    "gate_y_m": gy,
                    "gate_z_ned_m": float(gate[2]),
                    "min_distance_m": float(distance[closest]),
                    "closest_time_s": float(ds.df["time_s"].iloc[closest]),
                    "pass_count_lt_0p55m": entries,
                }
            )
    return pd.DataFrame(rows)


def calibration_factors(path: Path | None, metric_names: list[str]) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "domain" not in frame:
        return {}
    real = frame.loc[frame["domain"].astype(str).str.lower() == "real"]
    sitl = frame.loc[frame["domain"].astype(str).str.lower() == "sitl"]
    if real.empty or sitl.empty:
        return {}
    factors: dict[str, float] = {}
    for metric in metric_names:
        if metric not in frame:
            continue
        real_value = float(real.iloc[0][metric])
        sitl_value = float(sitl.iloc[0][metric])
        if np.isfinite(real_value) and np.isfinite(sitl_value) and abs(sitl_value) > 1e-9:
            factors[metric] = real_value / sitl_value
    return factors


def make_prediction(
    summary: pd.DataFrame,
    calibration_path: Path | None,
    hand_weight: float,
    motor_weight: float,
) -> pd.DataFrame:
    hand = summary.iloc[0]
    sitl = summary.iloc[1]
    specs = [
        ("speed_p95_mps", "水平速度 P95", "m/s", hand_weight, 0.25, 0.35, 0.0, None, "SITL 动力学为主，手持只修正任务激进度"),
        ("height_mean_m", "平均高度", "m", 0.10, 0.0, 0.30, 0.0, None, "手持高度由人决定，预测主要采用 SITL 高度"),
        ("roll_abs_p95_deg", "横滚绝对值 P95", "deg", hand_weight, 0.30, 4.0, 0.0, 60.0, "手持倾角只作为外扰/场景残差"),
        ("pitch_abs_p95_deg", "俯仰绝对值 P95", "deg", hand_weight, 0.30, 4.0, 0.0, 60.0, "手持倾角只作为外扰/场景残差"),
        ("motor_cmd_max_p95", "最大电机指令 P95", "norm", motor_weight, 0.50, 0.15, 0.0, 1.0, "加桨闭环后应从空载饱和回落；不同日志归一化可能不一致"),
        ("motor_any_high_sat_rate", "任一电机饱和占比", "ratio", motor_weight, 1.00, 0.05, 0.0, 1.0, "空载手持饱和不会原样延续，仅保留小权重风险残差"),
        ("gate_min_distance_median", "门点最近距离中位数", "m", hand_weight, 0.60, 0.08, 0.0, None, "假设真实闭环整体接近 SITL，叠加少量手持路径残差"),
    ]
    factors = calibration_factors(calibration_path, [item[0] for item in specs])
    rows: list[dict[str, float | str]] = []
    for metric, label, unit, weight, relative_uncertainty, absolute_uncertainty, lower_bound, upper_bound, note in specs:
        if metric not in summary or not np.isfinite(float(hand.get(metric, np.nan))) or not np.isfinite(float(sitl.get(metric, np.nan))):
            continue
        hand_value = float(hand[metric])
        sitl_value = float(sitl[metric])
        factor = float(factors.get(metric, 1.0))
        prior = sitl_value * factor
        center = prior + weight * (hand_value - sitl_value)
        center = max(lower_bound, center)
        if upper_bound is not None:
            center = min(upper_bound, center)
        uncertainty = max(abs(center) * relative_uncertainty, absolute_uncertainty)
        low = max(lower_bound, center - uncertainty)
        high = center + uncertainty
        if upper_bound is not None:
            high = min(upper_bound, high)
        delta = center - sitl_value
        if abs(delta) <= max(abs(sitl_value) * 0.05, 1e-6):
            direction = "接近 SITL"
        else:
            direction = f"较 SITL {'上升' if delta > 0 else '下降'} {abs(delta):.3f} {unit}"
        rows.append(
            {
                "metric": metric,
                "指标": label,
                "unit": unit,
                "hand_observed": hand_value,
                "sitl_baseline": sitl_value,
                "historical_real_to_sitl_factor": factor,
                "hand_residual_weight": weight,
                "predicted_center": center,
                "predicted_low": low,
                "predicted_high": high,
                "trend_vs_sitl": direction,
                "confidence": "低（单个历史实飞样本 + 非物理外推）",
                "interpretation": note,
            }
        )
    return pd.DataFrame(rows)


def make_risk_indicators(summary: pd.DataFrame) -> pd.DataFrame:
    hand = summary.iloc[0]
    sitl = summary.iloc[1]
    saturation = float(hand.get("motor_any_high_sat_rate", np.nan))
    motor_p95 = float(hand.get("motor_cmd_max_p95", np.nan))
    spread = float(hand.get("motor_spread_p95", np.nan))
    roll = float(hand.get("roll_abs_p95_deg", np.nan))
    roll_ref = float(sitl.get("roll_abs_p95_deg", np.nan))
    speed = float(hand.get("speed_p95_mps", np.nan))
    speed_ref = float(sitl.get("speed_p95_mps", np.nan))
    rows = [
        {
            "indicator": "空载控制饱和",
            "hand_value": saturation,
            "sitl_value": float(sitl.get("motor_any_high_sat_rate", np.nan)),
            "status": "高风险" if saturation >= 0.10 else "关注" if saturation >= 0.03 else "正常",
            "meaning": f"手持任一电机 >= {MOTOR_SATURATION:.2f} 的时间比例；空载受约束会放大该值",
        },
        {
            "indicator": "最大电机指令",
            "hand_value": motor_p95,
            "sitl_value": float(sitl.get("motor_cmd_max_p95", np.nan)),
            "status": "高风险" if motor_p95 >= MOTOR_SATURATION else "关注" if motor_p95 >= 0.80 else "正常",
            "meaning": "P95 接近 1 表示无桨手持闭环长期缺少可实现的姿态响应",
        },
        {
            "indicator": "四电机指令离散度",
            "hand_value": spread,
            "sitl_value": float(sitl.get("motor_spread_p95", np.nan)),
            "status": "高风险" if spread >= 0.80 else "关注" if spread >= 0.50 else "正常",
            "meaning": "P95 最大/最小电机指令差；手持倾斜和无桨均会放大",
        },
        {
            "indicator": "横滚场景偏差",
            "hand_value": roll,
            "sitl_value": roll_ref,
            "status": "关注" if roll > max(20.0, 1.5 * roll_ref) else "正常",
            "meaning": "手持横滚 P95 与 SITL 对比，反映人工姿态/外扰覆盖",
        },
        {
            "indicator": "速度场景偏差",
            "hand_value": speed,
            "sitl_value": speed_ref,
            "status": "关注" if speed > 1.25 * speed_ref else "正常",
            "meaning": "手持速度来自人工移动，只用于判断测试是否比 SITL 更激进",
        },
    ]
    return pd.DataFrame(rows)


def interpolate_columns(df: pd.DataFrame, time_grid: np.ndarray, columns: list[str]) -> np.ndarray:
    source_time = df["time_s"].to_numpy(dtype=float)
    return np.column_stack(
        [np.interp(time_grid, source_time, df[column].to_numpy(dtype=float)) for column in columns]
    )


def map_hand_motors_to_sitl_envelope(
    hand_motor: np.ndarray,
    sitl_motor: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    mapped = np.empty_like(hand_motor)
    rows: list[dict[str, float | int]] = []
    for motor in range(hand_motor.shape[1]):
        hand_q05, hand_q50, hand_q95 = np.percentile(hand_motor[:, motor], [5, 50, 95])
        sitl_q01, sitl_q05, sitl_q50, sitl_q95, sitl_q99 = np.percentile(
            sitl_motor[:, motor], [1, 5, 50, 95, 99]
        )
        hand_span = max(float(hand_q95 - hand_q05), 1e-6)
        sitl_span = float(sitl_q95 - sitl_q05)
        mapped[:, motor] = sitl_q50 + (hand_motor[:, motor] - hand_q50) / hand_span * sitl_span
        mapped[:, motor] = np.clip(mapped[:, motor], sitl_q01, sitl_q99)
        rows.append(
            {
                "motor": motor,
                "hand_q05": hand_q05,
                "hand_q50": hand_q50,
                "hand_q95": hand_q95,
                "sitl_q01": sitl_q01,
                "sitl_q05": sitl_q05,
                "sitl_q50": sitl_q50,
                "sitl_q95": sitl_q95,
                "sitl_q99": sitl_q99,
            }
        )
    return mapped, pd.DataFrame(rows)


def centered_moving_average(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples == 1:
        return values.copy()
    if samples % 2 == 0:
        samples += 1
    half = samples // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(samples) / samples, mode="valid")


def trajectory_stats(name: str, time_s: np.ndarray, x: np.ndarray, y: np.ndarray, speed: np.ndarray) -> dict[str, float | str]:
    path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    return {
        "dataset": name,
        "duration_s": float(time_s[-1] - time_s[0]),
        "path_xy_m": path,
        "speed_mean_mps": float(np.mean(speed)),
        "speed_p95_mps": float(np.percentile(speed, 95)),
        "speed_max_mps": float(np.max(speed)),
        "end_x_m": float(x[-1]),
        "end_y_m": float(y[-1]),
    }


def predict_trajectory_from_motor_commands(
    hand_df: pd.DataFrame,
    sitl_df: pd.DataFrame,
    calibration_path: Path | None,
    prediction_hz: float,
    response_gain: float,
    differential_weight: float,
    min_speed_scale: float,
    max_speed_scale: float,
    smoothing_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    motor_cols = [f"motor_{i}" for i in range(4)]
    if not all(column in hand_df and column in sitl_df for column in motor_cols):
        raise ValueError("Motor trajectory prediction needs motor_0..motor_3 in both logs")
    horizon_s = min(float(hand_df["time_s"].iloc[-1]), float(sitl_df["time_s"].iloc[-1]))
    if horizon_s <= 0:
        raise ValueError("Motor trajectory prediction has an empty common time window")
    prediction_hz = max(float(prediction_hz), 5.0)
    time_s = np.arange(0.0, horizon_s + 0.5 / prediction_hz, 1.0 / prediction_hz)
    time_s = time_s[time_s <= horizon_s]

    hand_motor = interpolate_columns(hand_df, time_s, motor_cols)
    sitl_motor = interpolate_columns(sitl_df, time_s, motor_cols)
    mapped_motor, mapping = map_hand_motors_to_sitl_envelope(hand_motor, sitl_motor)

    hand_collective = np.mean(hand_motor, axis=1)
    hand_differential = np.max(hand_motor, axis=1) - np.min(hand_motor, axis=1)
    mapped_collective = np.mean(mapped_motor, axis=1)
    mapped_differential = np.max(mapped_motor, axis=1) - np.min(mapped_motor, axis=1)
    sitl_collective = np.mean(sitl_motor, axis=1)
    sitl_differential = np.max(sitl_motor, axis=1) - np.min(sitl_motor, axis=1)

    mapped_activity = mapped_collective + differential_weight * mapped_differential
    sitl_activity = sitl_collective + differential_weight * sitl_differential
    activity_ratio = mapped_activity / np.maximum(sitl_activity, 0.03)
    activity_ratio = centered_moving_average(activity_ratio, round(smoothing_s * prediction_hz))

    calibration = calibration_factors(calibration_path, ["speed_p95_mps"])
    historical_speed_factor = float(calibration.get("speed_p95_mps", 1.0))
    motor_speed_scale = 1.0 + response_gain * (activity_ratio - 1.0)
    motor_speed_scale = np.clip(motor_speed_scale, min_speed_scale, max_speed_scale)
    motor_speed_scale *= historical_speed_factor

    sitl_x = interpolate_columns(sitl_df, time_s, ["pos_x_m"])[:, 0]
    sitl_y = interpolate_columns(sitl_df, time_s, ["pos_y_m"])[:, 0]
    sitl_speed = interpolate_columns(sitl_df, time_s, ["speed_xy_mps"])[:, 0]
    hand_x = interpolate_columns(hand_df, time_s, ["pos_x_m"])[:, 0]
    hand_y = interpolate_columns(hand_df, time_s, ["pos_y_m"])[:, 0]
    hand_speed = interpolate_columns(hand_df, time_s, ["speed_xy_mps"])[:, 0]

    predicted_speed = np.maximum(0.0, sitl_speed * motor_speed_scale)
    predicted_x = np.empty_like(sitl_x)
    predicted_y = np.empty_like(sitl_y)
    predicted_x[0] = sitl_x[0]
    predicted_y[0] = sitl_y[0]
    predicted_x[1:] = predicted_x[0] + np.cumsum(np.diff(sitl_x) * motor_speed_scale[:-1])
    predicted_y[1:] = predicted_y[0] + np.cumsum(np.diff(sitl_y) * motor_speed_scale[:-1])

    output = pd.DataFrame(
        {
            "time_s": time_s,
            "hand_x_m": hand_x,
            "hand_y_m": hand_y,
            "hand_speed_xy_mps": hand_speed,
            "sitl_x_m": sitl_x,
            "sitl_y_m": sitl_y,
            "sitl_speed_xy_mps": sitl_speed,
            "predicted_x_m": predicted_x,
            "predicted_y_m": predicted_y,
            "predicted_speed_xy_mps": predicted_speed,
            "hand_motor_collective_raw": hand_collective,
            "hand_motor_differential_raw": hand_differential,
            "mapped_motor_collective": mapped_collective,
            "mapped_motor_differential": mapped_differential,
            "sitl_motor_collective": sitl_collective,
            "sitl_motor_differential": sitl_differential,
            "motor_activity_ratio": activity_ratio,
            "motor_speed_scale": motor_speed_scale,
        }
    )
    for motor in range(4):
        output[f"hand_motor_{motor}_raw"] = hand_motor[:, motor]
        output[f"hand_motor_{motor}_mapped"] = mapped_motor[:, motor]
        output[f"sitl_motor_{motor}"] = sitl_motor[:, motor]

    stats_rows = [
        trajectory_stats("手持观测（人工运动）", time_s, hand_x, hand_y, hand_speed),
        trajectory_stats("SITL 同时间段", time_s, sitl_x, sitl_y, sitl_speed),
        trajectory_stats("当前电机指令加桨预测", time_s, predicted_x, predicted_y, predicted_speed),
    ]
    prediction_rmse = float(np.sqrt(np.mean((predicted_x - sitl_x) ** 2 + (predicted_y - sitl_y) ** 2)))
    final_error = float(np.hypot(predicted_x[-1] - sitl_x[-1], predicted_y[-1] - sitl_y[-1]))
    for row in stats_rows:
        row["xy_rmse_vs_sitl_m"] = prediction_rmse if row["dataset"] == "当前电机指令加桨预测" else float("nan")
        row["final_xy_error_vs_sitl_m"] = final_error if row["dataset"] == "当前电机指令加桨预测" else float("nan")
        row["historical_speed_factor"] = historical_speed_factor
        row["model_confidence"] = "低（SITL 路径方向 + 电机包线经验映射）"
    return output, pd.DataFrame(stats_rows), mapping


def plot_motor_trajectory_prediction(prediction: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.plot(prediction["sitl_x_m"], prediction["sitl_y_m"], color="#d62728", lw=1.5, label="SITL 同时间段")
    ax.plot(prediction["hand_x_m"], prediction["hand_y_m"], color="#1f77b4", lw=1.0, alpha=0.65, label="手持观测（人工）")
    ax.plot(prediction["predicted_x_m"], prediction["predicted_y_m"], color="#2ca02c", lw=1.8, label="当前电机指令加桨预测")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X 位置 (m)")
    ax.set_ylabel("Y 位置 (m)")
    ax.set_title("XY 轨迹预测")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(prediction["time_s"], prediction["sitl_speed_xy_mps"], color="#d62728", lw=1.2, label="SITL")
    ax.plot(prediction["time_s"], prediction["hand_speed_xy_mps"], color="#1f77b4", lw=0.9, alpha=0.65, label="手持观测")
    ax.plot(prediction["time_s"], prediction["predicted_speed_xy_mps"], color="#2ca02c", lw=1.5, label="电机指令预测")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("水平速度 (m/s)")
    ax.set_title("速度预测")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(prediction["time_s"], prediction["motor_activity_ratio"], color="#9467bd", lw=1.0, label="映射后电机活动/SITL")
    ax.plot(prediction["time_s"], prediction["motor_speed_scale"], color="#2ca02c", lw=1.2, label="最终速度倍率")
    ax.axhline(1.0, color="#666666", ls="--", lw=0.9)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("倍率")
    ax.set_title("电机指令推导的速度倍率")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    raw_max = prediction[[f"hand_motor_{i}_raw" for i in range(4)]].max(axis=1)
    mapped_max = prediction[[f"hand_motor_{i}_mapped" for i in range(4)]].max(axis=1)
    sitl_max = prediction[[f"sitl_motor_{i}" for i in range(4)]].max(axis=1)
    ax.plot(prediction["time_s"], raw_max, color="#1f77b4", lw=0.9, label="手持原始最大指令")
    ax.plot(prediction["time_s"], mapped_max, color="#2ca02c", lw=1.1, label="映射后最大指令")
    ax.plot(prediction["time_s"], sitl_max, color="#d62728", lw=0.9, label="SITL 最大指令")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("归一化指令")
    ax.set_title("空载指令到可飞包线的映射")
    ax.legend(fontsize=8)

    fig.suptitle("基于当前四电机指令的加桨轨迹/速度粗略预测", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = out / "18_motor_based_trajectory_speed_prediction_cn.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_prediction_dashboard(prediction: pd.DataFrame, out: Path) -> Path:
    shown = prediction.head(6).reset_index(drop=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (_, row) in zip(axes.ravel(), shown.iterrows()):
        center = float(row["predicted_center"])
        low = float(row["predicted_low"])
        high = float(row["predicted_high"])
        values = [float(row["sitl_baseline"]), float(row["hand_observed"]), center]
        ax.bar(["SITL", "手持", "加桨预测"], values, color=["#d62728", "#1f77b4", "#2ca02c"])
        ax.errorbar(2, center, yerr=[[center - low], [high - center]], fmt="none", color="black", capsize=5, lw=1.4)
        ax.set_title(f"{row['指标']} ({row['unit']})")
        ax.tick_params(axis="x", rotation=15)
    for ax in axes.ravel()[len(shown) :]:
        ax.axis("off")
    fig.suptitle("加桨实飞趋势筛查预测（误差棒为保守区间，非飞行动力学仿真）", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = out / "17_prop_flight_prediction_cn.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    datasets: list[Dataset],
    summary: pd.DataFrame,
    prediction: pd.DataFrame,
    motor_prediction_summary: pd.DataFrame,
    risks: pd.DataFrame,
    calibration_path: Path | None,
) -> Path:
    hand = summary.iloc[0]
    sitl = summary.iloc[1]
    high_risk = (risks["status"] == "高风险").any()
    recommendation = (
        "不建议直接进入完整自主赛道实飞。先做固定保护架/系留条件下的低功率响应检查，再做空旷隔离区低高度悬停。"
        if high_risk
        else "可进入受控、低高度、隔离区域的带桨验证，但本报告不构成放飞许可。"
    )
    motor_sitl = motor_prediction_summary.loc[motor_prediction_summary["dataset"] == "SITL 同时间段"].iloc[0]
    motor_predicted = motor_prediction_summary.loc[
        motor_prediction_summary["dataset"] == "当前电机指令加桨预测"
    ].iloc[0]
    lines = [
        f"# {datasets[0].name} 与 {datasets[1].name} 专项分析",
        "",
        "## 结论",
        "",
        f"- 当前结论：{recommendation}",
        f"- 手持与 SITL 的水平速度 P95 分别为 {hand['speed_p95_mps']:.3f} 和 {sitl['speed_p95_mps']:.3f} m/s；手持值由人工运动产生，不能当作真实飞行速度性能。",
        f"- 手持任一电机 >= {MOTOR_SATURATION:.2f} 的占比为 {hand['motor_any_high_sat_rate']:.1%}，SITL 为 {sitl['motor_any_high_sat_rate']:.1%}。无桨且被手约束时控制器无法靠推力消除误差，因此该饱和不能直接外推为带桨饱和，但它阻止我们仅凭手持日志确认实飞安全。",
        f"- 手持横滚绝对值 P95 为 {hand['roll_abs_p95_deg']:.2f}°，SITL 为 {sitl['roll_abs_p95_deg']:.2f}°；当前手持场景更激进。",
        "",
        "## 当前电机指令预测轨迹和速度",
        "",
        f"- 预测窗口为 {motor_predicted['duration_s']:.2f} s，与手持运动段和 SITL 的公共时间窗口一致。",
        f"- 当前四电机指令对应的加桨预测速度：平均 {motor_predicted['speed_mean_mps']:.3f} m/s，P95 {motor_predicted['speed_p95_mps']:.3f} m/s，最大 {motor_predicted['speed_max_mps']:.3f} m/s。",
        f"- 同时间段 SITL 的速度 P95 为 {motor_sitl['speed_p95_mps']:.3f} m/s；预测路径长度为 {motor_predicted['path_xy_m']:.3f} m，SITL 为 {motor_sitl['path_xy_m']:.3f} m。",
        f"- 预测轨迹相对同时间段 SITL 的时序 XY RMSE 为 {motor_predicted['xy_rmse_vs_sitl_m']:.3f} m，终点偏差 {motor_predicted['final_xy_error_vs_sitl_m']:.3f} m。",
        "",
        "计算方法：先将四路手持空载电机指令的 5%/50%/95% 分位数映射到 SITL 各通道的可飞指令包线；再用 `电机活动 = 四电机均值 + 差动权重 × (最大值-最小值)` 计算活动倍率。速度倍率为 `1 + 响应增益 × (活动倍率-1)`，经过限幅和平滑后乘到 SITL 速度；预测 XY 则把该倍率乘到 SITL 每个时刻的位移增量并积分。",
        "",
        "这个模型只用电机指令决定速度倍率和路径尺度，路径方向来自 SITL。原因是仅靠四路电机指令、没有带桨推力曲线、质量/惯量、姿态初值和风场时，无法唯一恢复全局 XY 方向。它适合看大体趋势，不是独立的飞行动力学预测。",
        "",
        "## 哪些数据可以比较",
        "",
        "- 可比较：轨迹形状、定位连续性、航向更新方向、任务速度量级、控制模式是否进入、通道是否缺失、指令是否异常顶满。",
        "- 不可直接比较：空载 RPM 与带桨 RPM/推力、手持位置误差与自主飞行跟踪误差、手持速度与真实飞行最高速度、空载电机饱和占比与带桨饱和占比。",
        "- 时序图使用各自完整运动段的归一化时间，只比较形态和节奏，不表示逐时刻同步。",
        "",
        "## 加桨趋势预测方法",
        "",
        "预测中心 = 当前 SITL × 历史实飞/SITL 比例 + 手持残差权重 ×（当前手持 − 当前 SITL）。",
        "运动/姿态默认权重 0.20，电机默认权重 0.10；区间是人为设定的保守筛查区间。该模型不是推力台模型或六自由度动力学仿真。",
        f"历史校准文件：`{calibration_path}`" if calibration_path else "未提供历史实飞校准，历史比例按 1.0 处理。",
        "",
        "| 指标 | SITL | 手持观测 | 加桨预测中心 | 预测区间 | 趋势 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in prediction.iterrows():
        lines.append(
            f"| {row['指标']} ({row['unit']}) | {row['sitl_baseline']:.3f} | {row['hand_observed']:.3f} | "
            f"{row['predicted_center']:.3f} | {row['predicted_low']:.3f}–{row['predicted_high']:.3f} | {row['trend_vs_sitl']} |"
        )
    lines.extend(
        [
            "",
            "## 风险筛查",
            "",
            "| 指标 | 手持 | SITL | 状态 | 含义 |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in risks.iterrows():
        lines.append(f"| {row['indicator']} | {row['hand_value']:.3f} | {row['sitl_value']:.3f} | {row['status']} | {row['meaning']} |")
    lines.extend(
        [
            "",
            "## 建议的带桨验证顺序",
            "",
            "1. 继续保持拆桨，确认电机顺序、旋向、失控保护和急停链路。",
            "2. 使用保护架或可靠系留，限制推力，检查控制方向是否正确；不要手持已装桨并解锁的飞行器。",
            "3. 在隔离、空旷区域做低高度短时悬停，记录第一份真正带载日志。",
            "4. 用 `compare_real_sitl_logs.py` 对第一份实飞日志复核电机饱和、姿态振荡、跟踪误差和 failsafe，再决定是否跑完整赛道。",
            "",
            "## 安全依据",
            "",
            "- Betaflight 的 Runaway Takeoff Prevention 文档明确说明：无桨手持移动时，飞行器不能响应 PID 修正，PID_sum 会继续增大，甚至触发解除武装；严禁手持已装桨并解锁的飞行器。",
            "- Betaflight Safety 文档要求：除非确实准备飞行，否则不要在装桨状态解锁；设置、刷写或存在疑问时应拆桨。",
            "- 官方链接：https://betaflight.com/docs/wiki/guides/current/Runaway-Takeoff-Prevention",
            "- 官方链接：https://betaflight.com/docs/wiki/guides/current/Safety",
            "",
            "## 输出",
            "",
            "- `tables/comparison_summary.csv`：普通统计指标。",
            "- `tables/gate_metrics.csv`：门点最近距离和进入次数。",
            "- `tables/prop_flight_prediction.csv`：预测公式输入、中心值和区间。",
            "- `tables/motor_trajectory_prediction.csv`：逐时刻的映射前/后电机指令、速度倍率、预测速度和预测 XY。",
            "- `tables/motor_trajectory_prediction_summary.csv`：预测路径长度、速度统计和相对 SITL 偏差。",
            "- `tables/motor_envelope_mapping.csv`：四路手持空载指令到 SITL 可飞包线的分位数映射参数。",
            "- `tables/risk_indicators.csv`：手持场景风险筛查。",
            "- `processed/`：截取后的标准化时序。",
            "- `figures/`：轨迹、速度、高度、姿态、电机和预测图。",
        ]
    )
    path = out_dir / "hand_sitl_analysis_cn.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_summary(summary: pd.DataFrame, tables_dir: Path) -> None:
    summary.to_csv(tables_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    with (tables_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dedicated handheld no-prop vs SITL analysis with a conservative prop-flight trend estimate.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON configuration path.")
    ap.add_argument("--hand-log", default=None, help="Handheld no-prop CSV path; overrides config.")
    ap.add_argument("--sitl-log", default=None, help="SITL CSV path; overrides config.")
    ap.add_argument("--out-root", default=None, help="Output root directory; overrides config.")
    ap.add_argument("--out-name", default=None, help="Output folder name; overrides config.")
    ap.add_argument("--max-plot-points", type=int, default=None, help="Downsample each series to this many points.")
    return ap.parse_args()


def main() -> int:
    configure_matplotlib()
    args = parse_args()
    config_path = resolve_workspace_path(args.config) if args.config else None
    config = load_config(config_path)
    hand_path = resolve_workspace_path(cfg_value(config, "hand_log", args.hand_log, DEFAULT_HAND)).resolve()
    sitl_path = resolve_workspace_path(cfg_value(config, "sitl_log", args.sitl_log, DEFAULT_SITL)).resolve()
    out_root = resolve_workspace_path(cfg_value(config, "out_root", args.out_root, DEFAULT_OUT_ROOT)).resolve()
    out_name = str(cfg_value(config, "out_name", args.out_name, "hand_vs_sitl"))
    hand_label = str(config.get("hand_label", "无桨手持"))
    sitl_label = str(config.get("sitl_label", "SITL"))
    hand_flags = tuple(int(v) for v in config.get("hand_control_flags", [12803, 12291]))
    speed_threshold = float(config.get("motion_speed_mps", 0.30))
    displacement_threshold = float(config.get("motion_displacement_m", 0.20))
    max_points = max(100, int(cfg_value(config, "max_plot_points", args.max_plot_points, MAX_PLOT_POINTS)))
    gates = normalize_gates(config.get("gates", DEFAULT_GATES_NED))
    gate_labels = normalize_gate_labels(config.get("gate_labels", DEFAULT_GATE_LABELS), len(gates))
    calibration_value = config.get("calibration_metrics")
    calibration_path = resolve_workspace_path(calibration_value).resolve() if calibration_value else None
    hand_weight = float(config.get("prediction_hand_residual_weight", 0.20))
    motor_weight = float(config.get("prediction_motor_residual_weight", 0.10))
    motor_prediction_hz = float(config.get("motor_trajectory_prediction_hz", 50.0))
    motor_response_gain = float(config.get("motor_trajectory_response_gain", 0.35))
    motor_differential_weight = float(config.get("motor_trajectory_differential_weight", 0.35))
    motor_min_speed_scale = float(config.get("motor_trajectory_min_speed_scale", 0.70))
    motor_max_speed_scale = float(config.get("motor_trajectory_max_speed_scale", 1.30))
    motor_smoothing_s = float(config.get("motor_trajectory_smoothing_s", 0.30))

    if not hand_path.exists():
        raise FileNotFoundError(hand_path)
    if not sitl_path.exists():
        raise FileNotFoundError(sitl_path)

    hand_full, hand_profile = load_hand_frame(hand_path)
    sitl_full, sitl_profile = load_sitl_frame(sitl_path)
    add_motor_omega_columns(hand_full, hand_path)
    add_motor_omega_columns(sitl_full, sitl_path)
    hand_df, hand_profile, _ = select_active_segment(
        hand_full, hand_profile, "hand", hand_flags, speed_threshold, displacement_threshold
    )
    sitl_df, sitl_profile, _ = select_active_segment(
        sitl_full, sitl_profile, "sitl", (), speed_threshold, displacement_threshold
    )
    add_aggregate_columns(hand_df)
    add_aggregate_columns(sitl_df)
    datasets = [
        Dataset(hand_label, str(config.get("hand_color", "#1f77b4")), hand_df, hand_profile),
        Dataset(sitl_label, str(config.get("sitl_color", "#d62728")), sitl_df, sitl_profile),
    ]

    out_dir = out_root / out_name
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    processed_dir = out_dir / "processed"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    passage_tables = [
        evaluate_gate_passages(ds.df, gates, gate_labels, ds.name)
        for ds in datasets
    ]

    figures = [
        plot_xy_side_by_side(datasets, figures_dir, max_points),
        plot_xy_overlay(datasets, figures_dir, max_points),
        plot_series_side_by_side(
            datasets,
            figures_dir,
            "height_m",
            "高度 (m)",
            "高度对比",
            "03_height_side_by_side_same_scale.png",
            extra_cols=("height_sp_m", "SITL高度设定"),
            max_points=max_points,
        ),
        plot_series_overlay(datasets, figures_dir, "height_m", "高度 (m)", "高度", "04_height_overlay_same_axis.png", max_points),
        plot_series_side_by_side(
            datasets,
            figures_dir,
            "speed_xy_mps",
            "水平速度 (m/s)",
            "水平速度对比",
            "05_speed_side_by_side_same_scale.png",
            max_points=max_points,
        ),
        plot_series_overlay(datasets, figures_dir, "speed_xy_mps", "水平速度 (m/s)", "水平速度", "06_speed_overlay_same_axis.png", max_points),
        plot_series_side_by_side(
            datasets,
            figures_dir,
            "yaw_deg",
            "航向角 (deg)",
            "航向角对比",
            "07_yaw_side_by_side_same_scale.png",
            max_points=max_points,
        ),
        plot_series_overlay(datasets, figures_dir, "yaw_deg", "航向角 (deg)", "航向角", "08_yaw_overlay_same_axis.png", max_points),
        plot_series_overlay(datasets, figures_dir, "roll_deg", "横滚角 (deg)", "横滚角", "09_roll_overlay_same_axis.png", max_points),
        plot_series_overlay(datasets, figures_dir, "pitch_deg", "俯仰角 (deg)", "俯仰角", "10_pitch_overlay_same_axis.png", max_points),
        plot_series_side_by_side(
            datasets,
            figures_dir,
            "omega_mean_rpm",
            "平均电机转速 (rpm)",
            "平均电机转速对比",
            "11_omega_mean_side_by_side_same_scale.png",
            max_points=max_points,
        ),
        plot_series_overlay(
            datasets,
            figures_dir,
            "omega_mean_rpm",
            "平均电机转速 (rpm)",
            "平均电机转速",
            "12_omega_mean_overlay_same_axis.png",
            max_points,
        ),
        plot_omega_channels_side_by_side(datasets, figures_dir, max_points),
        plot_omega_channels_overlay(datasets, figures_dir, max_points),
        plot_series_overlay(datasets, figures_dir, "motor_max", "归一化指令", "最大电机指令", "15_motor_max_overlay_same_axis.png", max_points),
        plot_series_overlay(datasets, figures_dir, "motor_spread", "归一化指令差", "四电机指令离散度", "16_motor_spread_overlay_same_axis.png", max_points),
        plot_trajectory_gate_3d(
            [(ds.name, ds.color, ds.df) for ds in datasets],
            passage_tables,
            gates,
            gate_labels,
            figures_dir / "19_trajectory_3d_with_gates.png",
            max_points,
            "手持与 SITL 三维轨迹及 1.5 m × 1.5 m 门框",
        ),
    ]
    summary = make_summary(datasets)
    gate_metrics = make_gate_metrics(datasets, gates, gate_labels)
    if not gate_metrics.empty:
        for ds in datasets:
            values = gate_metrics.loc[gate_metrics["dataset"] == ds.name, "min_distance_m"]
            summary.loc[summary["dataset"] == ds.name, "gate_min_distance_median"] = float(values.median())
    for ds, passage_table in zip(datasets, passage_tables):
        mask = summary["dataset"] == ds.name
        summary.loc[mask, "gates_passed_3d"] = int(passage_table["passed"].sum())
        summary.loc[mask, "gates_total_3d"] = int(len(passage_table))
    prediction = make_prediction(summary, calibration_path, hand_weight, motor_weight)
    risks = make_risk_indicators(summary)
    figures.append(plot_prediction_dashboard(prediction, figures_dir))
    motor_trajectory, motor_trajectory_summary, motor_mapping = predict_trajectory_from_motor_commands(
        hand_df,
        sitl_df,
        calibration_path,
        motor_prediction_hz,
        motor_response_gain,
        motor_differential_weight,
        motor_min_speed_scale,
        motor_max_speed_scale,
        motor_smoothing_s,
    )
    figures.append(plot_motor_trajectory_prediction(motor_trajectory, figures_dir))

    export_summary(summary, tables_dir)
    if not gate_metrics.empty:
        gate_metrics.to_csv(tables_dir / "gate_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(passage_tables, ignore_index=True).to_csv(
        tables_dir / "gate_passage_3d.csv", index=False, encoding="utf-8-sig"
    )
    prediction.to_csv(tables_dir / "prop_flight_prediction.csv", index=False, encoding="utf-8-sig")
    motor_trajectory.to_csv(tables_dir / "motor_trajectory_prediction.csv", index=False, encoding="utf-8-sig")
    motor_trajectory_summary.to_csv(
        tables_dir / "motor_trajectory_prediction_summary.csv", index=False, encoding="utf-8-sig"
    )
    motor_mapping.to_csv(tables_dir / "motor_envelope_mapping.csv", index=False, encoding="utf-8-sig")
    risks.to_csv(tables_dir / "risk_indicators.csv", index=False, encoding="utf-8-sig")
    hand_df.to_csv(processed_dir / "hand_selected.csv", index=False, encoding="utf-8-sig")
    sitl_df.to_csv(processed_dir / "sitl_selected.csv", index=False, encoding="utf-8-sig")
    report_path = write_report(
        out_dir, datasets, summary, prediction, motor_trajectory_summary, risks, calibration_path
    )

    manifest_rows = [{"kind": "figure", "path": str(path.relative_to(out_dir))} for path in figures]
    manifest_rows.extend(
        {"kind": "table", "path": str(path.relative_to(out_dir))}
        for path in sorted(tables_dir.glob("*"))
    )
    manifest_rows.extend(
        {"kind": "processed", "path": str(path.relative_to(out_dir))}
        for path in sorted(processed_dir.glob("*"))
    )
    manifest_rows.append({"kind": "report", "path": str(report_path.relative_to(out_dir))})
    pd.DataFrame(manifest_rows).to_csv(out_dir / "deliverable_manifest.csv", index=False, encoding="utf-8-sig")

    print(f"Output directory: {out_dir}")
    print("Generated figures:")
    for figure in figures:
        print(f"  {figure}")
    print(summary.to_string(index=False))
    print("\nProp-flight trend prediction:")
    print(prediction[["指标", "sitl_baseline", "hand_observed", "predicted_center", "predicted_low", "predicted_high"]].to_string(index=False))
    print("\nMotor-command trajectory/speed prediction:")
    print(motor_trajectory_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
