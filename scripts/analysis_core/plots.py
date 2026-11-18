"""PNG plots shared by all seven analysis entry points."""
from __future__ import annotations

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .track_3d import plot_trajectory_gate_3d

COLORS = {"SITL": "#D55E00", "HIL": "#0072B2", "REAL": "#009E73"}
MAX_PLOT_POINTS = 7000


def setup_plots():
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": .25,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def sampled(frame):
    indices = np.unique(np.linspace(0, len(frame) - 1, min(len(frame), MAX_PLOT_POINTS), dtype=int))
    return frame.iloc[indices]


def save_figure(fig, path):
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    path.write_bytes(buffer.getvalue())
    plt.close(fig)


def time_plot(datasets, columns, output, title, duration, time_label="各自分析段起点后的时间 (s)"):
    fig, axes = plt.subplots(len(columns), len(datasets), figsize=(6 * len(datasets), 2.8 * len(columns)),
                             squeeze=False, sharex=True, sharey="row")
    for j, (name, frame) in enumerate(datasets):
        data = sampled(frame)
        for i, (column, label) in enumerate(columns):
            axes[i, j].plot(data.time_s, data[column], color=COLORS[name], lw=.9)
            axes[i, j].set(xlim=(0, duration), ylabel=label)
            if i == 0:
                axes[i, j].set_title(name)
            if i == len(columns) - 1:
                axes[i, j].set_xlabel(time_label)
    fig.suptitle(title)
    save_figure(fig, output)


def xy_plot(datasets, gates, output):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, frame in datasets:
        data = sampled(frame)
        ax.plot(data.pos_x_m, data.pos_y_m, color=COLORS[name], lw=1, alpha=.85, label=name)
        ax.scatter(data.pos_x_m.iloc[0], data.pos_y_m.iloc[0], color=COLORS[name], s=35)
        ax.scatter(data.pos_x_m.iloc[-1], data.pos_y_m.iloc[-1], color=COLORS[name], marker="x", s=35)
    ax.scatter(gates[:, 0], gates[:, 1], color="black", marker="+", label="门中心")
    for i, gate in enumerate(gates, 1):
        ax.annotate(str(i), gate[:2], xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="XY 轨迹（圆点为起点，叉号为终点）")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    save_figure(fig, output)


def gate_errors_plot(passages, output):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, passage in passages:
        valid = passage.loc[passage.passed]
        for ax, column in zip(axes, ("lateral_error_m", "vertical_error_m")):
            ax.plot(valid.gate_index, valid[column], color=COLORS[name], marker="o", label=name)
    for ax, label in zip(axes, ("横向偏差 (m)", "竖向偏差 (m, NED)")):
        ax.axhline(0, color="black", lw=.7)
        ax.set_ylabel(label)
        ax.legend()
    axes[-1].set_xlabel("门序号")
    fig.suptitle("首组有序过门交点偏差")
    save_figure(fig, output)


def motor_plot(items, prefix, output, duration):
    if not any(any(f"{prefix}_{i}" in item["window"] for i in range(4)) for item in items):
        return False
    fig, axes = plt.subplots(2, len(items), figsize=(6 * len(items), 7), squeeze=False, sharex=True)
    group = "command_columns" if prefix == "command" else "rotation_columns"
    for j, item in enumerate(items):
        data, profile = sampled(item["window"]), item["profile"]
        columns = [f"{prefix}_{i}" for i in range(4) if f"{prefix}_{i}" in data]
        for column in columns:
            source = profile["signal_metadata"][column][0]
            axes[0, j].plot(data.time_s, data[column], lw=.8, label=source)
        if columns:
            axes[0, j].legend(ncol=2, fontsize=8)
        else:
            axes[0, j].text(.5, .5, "日志无该信号", transform=axes[0, j].transAxes, ha="center")
        if len(columns) == 4:
            four = data[columns].where(np.isfinite(data[columns]))
            valid = four.notna().all(axis=1)
            axes[1, j].plot(data.loc[valid, "time_s"], four.loc[valid].mean(axis=1), label="四路均值", lw=.9)
            axes[1, j].plot(data.loc[valid, "time_s"], four.loc[valid].max(axis=1) - four.loc[valid].min(axis=1),
                            label="四路极差", lw=.9)
            axes[1, j].legend()
        else:
            axes[1, j].text(.5, .5, "四路信号不足", transform=axes[1, j].transAxes, ha="center")
        axes[0, j].set_title(profile["dataset"] + " · " + ", ".join(profile[group]))
        axes[0, j].set_ylabel("日志原值")
        axes[1, j].set(ylabel="均值 / 极差（原值）", xlabel="各自分析段起点后的时间 (s)")
        for ax in axes[:, j]:
            ax.set_xlim(0, duration)
    fig.suptitle("电机指令原值" if prefix == "command" else "电机转速信号原值")
    save_figure(fig, output)
    return True


def distributions_plot(datasets, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (column, label) in zip(axes.flat, (
        ("speed_xy_mps", "水平速度 (m/s)"), ("height_m", "高度 (m)"),
        ("roll_deg", "Roll (°)"), ("pitch_deg", "Pitch (°)"),
    )):
        values = np.concatenate([frame[column].to_numpy() for _, frame in datasets])
        edges = np.histogram_bin_edges(values, bins=60)
        for name, frame in datasets:
            ax.hist(frame[column], bins=edges, density=True, histtype="step", color=COLORS[name], label=name)
        ax.set(xlabel=label, ylabel="样本概率密度")
        ax.legend()
    fig.suptitle("分析窗口内信号分布")
    save_figure(fig, output)


def render_figures(items, gates, labels, output, duration):
    setup_plots()
    datasets = [(item["profile"]["dataset"], item["window"]) for item in items]
    passages = [(item["profile"]["dataset"], item["passages"]) for item in items]
    figures = []

    def record(filename, title):
        figures.append((filename, title))
        return output / filename

    xy_plot(datasets, gates, record("01_trajectory_xy.png", "XY 轨迹"))
    plot_trajectory_gate_3d(
        [(name, COLORS[name], frame) for name, frame in datasets], [p for _, p in passages],
        gates, labels, record("02_trajectory_3d_gates.png", "三维轨迹与门框"), MAX_PLOT_POINTS, "NED 三维轨迹与有序门",
    )
    time_groups = (
        ("03_position.png", "位置分量", [(f"pos_{a}_m", f"{a.upper()} NED (m)") for a in "xyz"]),
        ("04_velocity.png", "速度分量", [(f"vel_{a}_mps", f"V{a} NED (m/s)") for a in "xyz"]),
        ("05_speed_height.png", "速度与高度", [("speed_xy_mps", "水平速度 (m/s)"),
                                                 ("speed_3d_mps", "三维速度 (m/s)"), ("height_m", "高度 (m)")]),
        ("06_attitude.png", "姿态角", [(f"{a.lower()}_deg", f"{a} (°)") for a in ("Roll", "Pitch", "Yaw")]),
        ("07_yaw_unwrapped.png", "连续航向角", [("yaw_unwrapped_deg", "连续 Yaw (°)")]),
    )
    for filename, title, columns in time_groups:
        time_plot(datasets, columns, record(filename, title), title, duration)
    gate_errors_plot(passages, record("08_gate_errors.png", "过门位置偏差"))
    for prefix, filename, title in (("command", "09_motor_commands.png", "四路电机指令"),
                                     ("omega", "10_motor_rotation.png", "四路转速信号")):
        if motor_plot(items, prefix, output / filename, duration):
            figures.append((filename, title))
    distributions_plot(datasets, record("11_distributions.png", "信号分布"))
    if all(item["gate_window"] is not None for item in items):
        local = [(item["profile"]["dataset"], item["gate_window"]) for item in items]
        end = min(frame.time_s.iloc[-1] for _, frame in local)
        local = [(name, frame.loc[frame.time_s <= end]) for name, frame in local]
        if all(len(frame) >= 2 for _, frame in local):
            for number, title, columns in (
                (12, "首门至末门位置", [(f"pos_{a}_m", f"{a.upper()} NED (m)") for a in "xyz"]),
                (13, "首门至末门速度", [(f"vel_{a}_mps", f"V{a} NED (m/s)") for a in "xyz"]),
                (14, "首门至末门姿态", [(f"{a.lower()}_deg", f"{a} (°)") for a in ("Roll", "Pitch", "Yaw")]),
            ):
                time_plot(local, columns, record(f"{number}_gate_window.png", title), title, end,
                          "各自首门事件后的时间 (s)")
    return figures
