"""Shared loading, geometry, and summaries for HIL and HIL/SITL analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_sitl_log import configure_matplotlib, quat_to_euler_rad
from track_3d import evaluate_gate_passages

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "hil_sitl_kidney.json"
MAX_PLOT_POINTS = 7000


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path) -> tuple[dict, np.ndarray, list[str]]:
    config = json.loads(resolve_path(path).read_text(encoding="utf-8-sig"))
    track = json.loads(resolve_path(config["track_config"]).read_text(encoding="utf-8-sig"))
    gates = np.asarray(track["gates"], dtype=float)
    if gates.ndim != 2 or gates.shape[1] != 4 or len(gates) == 0 or not np.isfinite(gates).all():
        raise ValueError("赛道 gates 必须是非空 [x, y, z(NED), yaw(度)] 数组")
    gates = gates.copy()
    gates[:, 3] = np.deg2rad(gates[:, 3])
    return config, gates, [f"Gate {i:02d}" for i in range(1, len(gates) + 1)]


def load_frame(path: Path, kind: str) -> tuple[pd.DataFrame, dict]:
    """Read HIL truth, SITL, or flight CSV using explicit signal sources."""
    if kind not in ("HIL", "SITL", "REAL"):
        raise ValueError(f"未知数据类型: {kind}")
    position = "extPos" if kind == "SITL" else "pos"
    velocity = "extVel" if kind == "SITL" else "vel"
    command = "command" if kind == "HIL" else "motor"
    header = pd.read_csv(path, nrows=0).columns
    time_col = "timeS" if "timeS" in header else "timeUs"
    required = [time_col] + [f"{p}[{i}]" for p in (position, velocity) for i in range(3)]
    required += [f"quat[{i}]" for i in range(4)]
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"{path}: 缺少必要字段: {', '.join(missing)}")
    optional = [f"{p}[{i}]" for p in (command, "omega") for i in range(4)]
    if kind == "REAL":
        optional.append("flightModeFlags")
    raw = pd.read_csv(path, usecols=required + [c for c in optional if c in header])
    raw = raw.apply(pd.to_numeric, errors="coerce")
    if len(raw) < 2 or not np.isfinite(raw[required].to_numpy()).all():
        raise ValueError(f"{path}: 必要字段包含无效值或记录不足两行")
    time = raw[time_col].to_numpy(dtype=float) / (1_000_000.0 if time_col == "timeUs" else 1.0)
    if not np.all(np.diff(time) > 0):
        raise ValueError(f"{path}: {time_col} 必须严格递增；请先检查时间重置或重复")
    quat = raw[[f"quat[{i}]" for i in range(4)]].to_numpy(dtype=float)
    if np.any(np.linalg.norm(quat, axis=1) < 1e-9):
        raise ValueError(f"{path}: 存在无效的零四元数")

    frame = pd.DataFrame({"time_s": time - time[0]})
    for i, axis in enumerate("xyz"):
        frame[f"pos_{axis}_m"] = raw[f"{position}[{i}]"]
        frame[f"vel_{axis}_mps"] = raw[f"{velocity}[{i}]"]
    frame["height_m"] = -frame["pos_z_m"]
    frame["speed_xy_mps"] = np.hypot(frame["vel_x_mps"], frame["vel_y_mps"])
    frame["speed_3d_mps"] = np.sqrt(sum(frame[f"vel_{axis}_mps"] ** 2 for axis in "xyz"))
    euler = np.rad2deg(quat_to_euler_rad(quat))
    for i, axis in enumerate(("roll", "pitch", "yaw")):
        frame[f"{axis}_deg"] = euler[:, i]
    for source, target in ((command, "command"), ("omega", "omega")):
        for i in range(4):
            col = f"{source}[{i}]"
            if col in raw:
                frame[f"{target}_{i}"] = raw[col]
    if "flightModeFlags" in raw:
        frame["flight_mode_flags"] = raw["flightModeFlags"]
    profile = {
        "dataset": kind, "source": str(path.resolve()), "rows": len(frame),
        "original_time_start_s": float(time[0]), "duration_s": float(time[-1] - time[0]),
        "time_source": time_col,
        "median_sample_interval_s": float(np.median(np.diff(time))),
        "max_sample_interval_s": float(np.max(np.diff(time))),
        "position_source": position, "velocity_source": velocity,
        "attitude_source": "quat [w,x,y,z] (assumed)",
        "command_source": command, "omega_source": "omega",
        "coordinate_assumption": "NED, height = -z; meters and seconds",
        "motor_units_and_channel_mapping": "unverified; raw values only",
    }
    return frame, profile


def select_real_control_segment(frame: pd.DataFrame, profile: dict, flags: list[int]) -> tuple[pd.DataFrame, dict]:
    """Select one contiguous control segment without joining separate flights."""
    if "flight_mode_flags" not in frame:
        raise ValueError("实飞日志缺少 flightModeFlags，无法按控制模式截取")
    if not flags:
        raise ValueError("控制模式 flags 不可为空")
    indices = np.flatnonzero(frame.flight_mode_flags.isin(flags).to_numpy())
    if len(indices) < 2:
        raise ValueError(f"实飞日志中匹配 {flags} 的控制样本不足")
    if np.any(np.diff(indices) != 1):
        raise ValueError(f"实飞日志中匹配 {flags} 的控制样本不连续，请先明确要使用的飞行段")
    start, end = float(frame.time_s.iloc[indices[0]]), float(frame.time_s.iloc[indices[-1]])
    selected = frame.iloc[indices].copy().reset_index(drop=True)
    selected["time_s"] -= start
    result_profile = dict(profile)
    result_profile.update({"selection": "flightModeFlags", "control_flags": [int(v) for v in flags],
                           "full_log_duration_s": profile["duration_s"], "full_log_rows": profile["rows"],
                           "selection_start_s": start, "selection_end_s": end,
                           "duration_s": end - start, "rows": len(selected)})
    return selected, result_profile


def truncate_for_plot(frame: pd.DataFrame, duration_s: float) -> pd.DataFrame:
    """Use recorded samples within [0, duration_s], without inventing boundary data."""
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("绘图时长必须为正的有限数值")
    result = frame.loc[frame.time_s <= duration_s].copy()
    if len(result) < 2:
        raise ValueError("绘图窗口内少于两个样本")
    return result


def summarize(frame: pd.DataFrame, profile: dict, passages: pd.DataFrame) -> dict:
    points = frame[[f"pos_{a}_m" for a in "xyz"]].to_numpy()
    dpos = np.diff(points, axis=0)
    return {
        **profile,
        "path_length_xy_m": float(np.linalg.norm(dpos[:, :2], axis=1).sum()),
        "speed_xy_mean_mps": float(frame.speed_xy_mps.mean()),
        "speed_xy_p95_mps": float(frame.speed_xy_mps.quantile(0.95)),
        "speed_xy_max_mps": float(frame.speed_xy_mps.max()),
        "height_mean_m": float(frame.height_m.mean()),
        "height_min_m": float(frame.height_m.min()),
        "height_max_m": float(frame.height_m.max()),
        "ordered_gates_passed": int(passages.passed.sum()),
        "ordered_gates_forward": int((passages.crossing_direction == "forward").sum()),
        "ordered_gates_total": len(passages),
    }


def passages_for(frame: pd.DataFrame, gates: np.ndarray, labels: list[str], name: str) -> pd.DataFrame:
    return evaluate_gate_passages(frame, gates, labels, name)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def clear_optional_outputs(figures: Path, tables: Path, extra_figures: tuple[str, ...] = ()) -> None:
    """Remove script-owned conditional outputs before a same-name rerun."""
    names = ("08_motor_commands.png", "09_motor_omega.png", "12_gate_window_position.png",
             "13_gate_window_velocity.png", "14_gate_window_attitude.png",
             "15_gate_window_motor_commands.png", "16_gate_window_motor_omega.png") + extra_figures
    for name in names:
        (figures / name).unlink(missing_ok=True)
    (tables / "gate_window_statistics.csv").unlink(missing_ok=True)


def sampled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[np.unique(np.linspace(0, len(frame) - 1, min(len(frame), MAX_PLOT_POINTS), dtype=int))]


def setup_plots() -> None:
    configure_matplotlib()


def xy_plot(datasets: list[tuple[str, pd.DataFrame]], gates: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, frame in datasets:
        d = sampled(frame)
        ax.plot(d.pos_x_m, d.pos_y_m, lw=1, alpha=.8, label=name)
        ax.scatter(d.pos_x_m.iloc[0], d.pos_y_m.iloc[0], s=30, marker="o")
    ax.scatter(gates[:, 0], gates[:, 1], color="black", marker="x", label="门中心")
    for i, gate in enumerate(gates, 1):
        ax.annotate(str(i), (gate[0], gate[1]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="XY 轨迹（圆点为所绘轨迹起点）")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def time_plot(datasets: list[tuple[str, pd.DataFrame]], columns: list[tuple[str, str]], output: Path, title: str,
              time_label: str = "各自分析段起点后的时间 (s)", x_end: float | None = None) -> None:
    # Each dataset retains its own independent time origin; no synchronization is implied.
    fig, axes = plt.subplots(len(columns), len(datasets), figsize=(7 * len(datasets), 2.8 * len(columns)), squeeze=False)
    for j, (name, frame) in enumerate(datasets):
        d = sampled(frame)
        for i, (column, ylabel) in enumerate(columns):
            axes[i, j].plot(d.time_s, d[column], lw=.9)
            if x_end is not None:
                axes[i, j].set_xlim(0, x_end)
            axes[i, j].set_ylabel(ylabel)
            if i == 0:
                axes[i, j].set_title(name)
            if i == len(columns) - 1:
                axes[i, j].set_xlabel(time_label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def motor_plot(frame: pd.DataFrame, output: Path, x_end: float | None = None) -> bool:
    columns = [c for c in frame.columns if c.startswith(("command_", "omega_"))]
    if not columns:
        return False
    d = sampled(frame)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, prefix in enumerate(("command", "omega")):
        for col in columns:
            if col.startswith(prefix + "_"):
                axes[i].plot(d.time_s, d[col], lw=.8, label=col)
        axes[i].set_ylabel(f"{prefix}（原始数值）")
        if any(c.startswith(prefix + "_") for c in columns):
            axes[i].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("日志起点后的时间 (s)")
    if x_end is not None:
        axes[-1].set_xlim(0, x_end)
    fig.suptitle("HIL 电机信号（单位和通道映射待确认）")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return True


def parameter_table(frame: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Full-resolution sample statistics, preserving the original signal source."""
    signals = [(f"pos_{a}_m", "位置", "m", f"{profile['position_source']}[{i}]") for i, a in enumerate("xyz")]
    signals += [(f"vel_{a}_mps", "速度", "m/s", f"{profile['velocity_source']}[{i}]") for i, a in enumerate("xyz")]
    signals += [("height_m", "高度", "m", "-z"), ("speed_xy_mps", "速度", "m/s", "水平速度"),
                ("speed_3d_mps", "速度", "m/s", "三维速度")]
    signals += [(f"{axis}_deg", "姿态", "deg", "quat") for axis in ("roll", "pitch", "yaw")]
    signals += [(f"{name}_{i}", "电机", "原始值", f"{source}[{i}]")
                for name, source in (("command", profile["command_source"]), ("omega", profile["omega_source"]))
                for i in range(4)]
    values_by_name = {col: frame[col] for col, _, _, _ in signals if col in frame}
    for name, source in (("command", profile["command_source"]), ("omega", profile["omega_source"])):
        cols = [f"{name}_{i}" for i in range(4)]
        if all(col in frame for col in cols):
            four = frame[cols].where(np.isfinite(frame[cols]))
            complete = four.notna().all(axis=1)
            for suffix, series in (("mean", four.mean(axis=1)), ("spread", four.max(axis=1) - four.min(axis=1))):
                col = f"{name}_{suffix}"
                values_by_name[col] = series.where(complete)
                signals.append((col, "电机", "原始值", f"{source}[0..3] {suffix}"))
    rows = []
    for col, group, unit, source in signals:
        if col not in values_by_name:
            continue
        values = values_by_name[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        rows.append({"dataset": profile["dataset"], "group": group, "signal": col, "source": source,
                     "unit": unit, "valid_count": len(values), "missing_count": len(frame) - len(values),
                     "mean": float(np.mean(values)), "std": float(np.std(values)),
                     "min": float(np.min(values)), "p05": float(np.percentile(values, 5)),
                     "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                     "max": float(np.max(values)), "abs_p95": float(np.percentile(np.abs(values), 95))})
    return pd.DataFrame(rows)


def motor_channels_plot(datasets: list[tuple[str, pd.DataFrame]], prefix: str, output: Path,
                        time_label: str = "各自分析段起点后的时间 (s)", x_end: float | None = None) -> bool:
    """Plot each run independently; no cross-domain channel mapping is assumed."""
    if not any(any(f"{prefix}_{i}" in frame for i in range(4)) for _, frame in datasets):
        return False
    fig, axes = plt.subplots(2, len(datasets), figsize=(7 * len(datasets), 7), squeeze=False)
    for j, (name, frame) in enumerate(datasets):
        d = sampled(frame)
        columns = [f"{prefix}_{i}" for i in range(4) if f"{prefix}_{i}" in d]
        if not columns:
            axes[0, j].text(.5, .5, "无该信号", transform=axes[0, j].transAxes, ha="center")
        for col in columns:
            axes[0, j].plot(d.time_s, d[col], lw=.8, label=col)
        if columns:
            axes[0, j].legend(ncol=2, fontsize=8)
        if len(columns) == 4:
            complete = d[columns].where(np.isfinite(d[columns])).dropna()
            axes[1, j].plot(d.loc[complete.index, "time_s"], complete.mean(axis=1), label="四路均值", lw=.9)
            axes[1, j].plot(d.loc[complete.index, "time_s"], complete.max(axis=1) - complete.min(axis=1),
                            label="四路极差", lw=.9)
            axes[1, j].legend(fontsize=8)
        else:
            axes[1, j].text(.5, .5, "四路不完整，不计算均值和极差", transform=axes[1, j].transAxes, ha="center")
        axes[0, j].set_title(name)
        axes[0, j].set_ylabel(f"{prefix}（原始值）")
        axes[1, j].set_ylabel("均值 / 极差（原始值）")
        axes[1, j].set_xlabel(time_label)
        if x_end is not None:
            for ax in axes[:, j]:
                ax.set_xlim(0, x_end)
    fig.suptitle(f"{prefix} 四通道（单位、跨数据通道映射待确认；各自计时）")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return True


def matched_gate_window(frame: pd.DataFrame, passages: pd.DataFrame) -> pd.DataFrame | None:
    """Select recorded samples between the first and last matched forward gates."""
    if not passages.passed.all() or not (passages.crossing_direction == "forward").all():
        return None
    start, end = passages.passage_time_s.iloc[[0, -1]].to_numpy(dtype=float)
    if end <= start:
        return None
    window = frame.loc[frame.time_s.between(start, end)].copy()
    if len(window) < 2:
        return None
    window["time_s"] -= start
    return window


def gate_window_plots(datasets: list[tuple[str, pd.DataFrame]], output_dir: Path) -> None:
    """Show independent gate-anchored runs; equal time does not mean equal progress."""
    duration = min(float(frame.time_s.iloc[-1]) for _, frame in datasets)
    datasets = [(name, truncate_for_plot(frame, duration)) for name, frame in datasets]
    time_label = "各自首门事件后的时间 (s)"
    suffix = "首门至末门，按各自首门计时"
    time_plot(datasets, [("pos_x_m", "X (m)"), ("pos_y_m", "Y (m)"), ("pos_z_m", "Z NED (m)")],
              output_dir / "12_gate_window_position.png", "位置：" + suffix, time_label, duration)
    time_plot(datasets, [("vel_x_mps", "Vx (m/s)"), ("vel_y_mps", "Vy (m/s)"), ("vel_z_mps", "Vz NED (m/s)")],
              output_dir / "13_gate_window_velocity.png", "速度：" + suffix, time_label, duration)
    time_plot(datasets, [("roll_deg", "Roll (°)"), ("pitch_deg", "Pitch (°)"), ("yaw_deg", "Yaw (°)")],
              output_dir / "14_gate_window_attitude.png", "姿态：" + suffix, time_label, duration)
    motor_channels_plot(datasets, "command", output_dir / "15_gate_window_motor_commands.png", time_label, duration)
    motor_channels_plot(datasets, "omega", output_dir / "16_gate_window_motor_omega.png", time_label, duration)


def distributions_plot(datasets: list[tuple[str, pd.DataFrame]], output: Path) -> None:
    signals = [("speed_xy_mps", "水平速度 (m/s)"), ("height_m", "高度 (m)"),
               ("roll_deg", "Roll (°)"), ("pitch_deg", "Pitch (°)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (column, label) in zip(axes.flat, signals):
        for name, frame in datasets:
            ax.hist(frame[column], bins=60, density=True, histtype="step", lw=1.2, label=name)
        ax.set(xlabel=label, ylabel="概率密度")
        ax.legend(fontsize=8)
    fig.suptitle("绘图窗口内样本分布（含起飞）")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def gate_errors_plot(datasets: list[tuple[str, pd.DataFrame]], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, passage in datasets:
        valid = passage[passage.passed]
        for ax, col in zip(axes, ("lateral_error_m", "vertical_error_m")):
            ax.plot(valid.gate_index, valid[col], marker="o", lw=1, label=name)
    for ax, label in zip(axes, ("横向偏差 (m)", "竖向偏差 (m, NED)")):
        ax.axhline(0, color="black", lw=.7)
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("门序号")
    fig.suptitle("已匹配过门交点相对门中心偏差（含反向事件）")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
