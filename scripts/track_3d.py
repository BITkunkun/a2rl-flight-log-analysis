from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


GATE_WIDTH_M = 1.5
GATE_HEIGHT_M = 1.5
DEFAULT_GATES_NED = np.array(
    [
        [4.0, 0.0, -1.50],
        [2.0, 2.2, -1.50],
        [-1.1, 2.2, -1.50],
        [-5.0, 0.0, -1.50],
        [-1.1, -2.2, -1.50],
        [2.0, -2.2, -1.50],
        [4.0, 0.0, -1.50],
        [2.0, 2.2, -1.50],
        [-1.1, 2.2, -1.50],
        [-5.0, 0.0, -4.17],
        [-1.1, -2.2, -1.50],
        [2.0, -2.2, -1.50],
    ],
    dtype=float,
)
DEFAULT_GATE_LABELS = [f"Gate {idx:02d}" for idx in range(1, len(DEFAULT_GATES_NED) + 1)]


def normalize_gates(gates: np.ndarray | list[list[float]]) -> np.ndarray:
    result = np.asarray(gates, dtype=float)
    if result.ndim != 2 or result.shape[1] not in (2, 3, 4):
        raise ValueError("gates must be an N x 2, N x 3 or N x 4 array")
    if result.shape[1] == 2:
        result = np.column_stack([result, np.full(len(result), -1.5)])
    elif result.shape[1] == 4:
        result = result[:, :3]
    return result


def gate_normals(gates: np.ndarray | list[list[float]]) -> np.ndarray:
    """Gate-plane normals: explicit yaw column (N x 4 [x,y,z,yaw]) if present, else inferred from adjacent centers."""
    raw = np.asarray(gates, dtype=float)
    centers = normalize_gates(gates)
    if raw.ndim == 2 and raw.shape[1] == 4:
        yaw = raw[:, 3]
        return np.column_stack([np.cos(yaw), np.sin(yaw)])
    return gate_normals_xy(centers)


def normalize_gate_labels(labels: list[str], count: int) -> list[str]:
    result = [str(item) for item in labels[:count]]
    result.extend(f"Gate {idx:02d}" for idx in range(len(result) + 1, count + 1))
    return result


def gate_normals_xy(gates: np.ndarray) -> np.ndarray:
    """Estimate gate-plane normals from the adjacent centers in sequence."""
    centers = normalize_gates(gates)[:, :2]
    tangent = np.roll(centers, -1, axis=0) - np.roll(centers, 1, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm < 1e-9):
        raise ValueError("Cannot infer gate orientation from coincident adjacent gate centers")
    return tangent / norm[:, None]


def evaluate_gate_passages(
    df: pd.DataFrame,
    gates: np.ndarray | list[list[float]],
    labels: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    """Match ordered 3D gate-plane intersections inside the rectangular aperture."""
    centers = normalize_gates(gates)
    gate_labels = normalize_gate_labels(labels, len(centers))
    normals = gate_normals(gates)
    required = ["pos_x_m", "pos_y_m", "pos_z_m"]
    if any(col not in df for col in required):
        raise ValueError(f"3D gate analysis requires columns: {required}")
    time_col = "segment_time_s" if "segment_time_s" in df else "time_s"
    points = df[required].to_numpy(dtype=float)
    times = df[time_col].to_numpy(dtype=float)
    cursor = 0
    rows: list[dict[str, float | int | str | bool]] = []

    for gate_index, (center, normal, label) in enumerate(zip(centers, normals, gate_labels), start=1):
        signed = (points[:, :2] - center[:2]) @ normal
        finite_segments = (
            np.isfinite(signed[:-1])
            & np.isfinite(signed[1:])
            & np.all(np.isfinite(points[:-1]), axis=1)
            & np.all(np.isfinite(points[1:]), axis=1)
        )
        candidates = np.flatnonzero(
            finite_segments
            & (signed[:-1] * signed[1:] <= 0.0)
            & (np.abs(signed[1:] - signed[:-1]) > 1e-9)
        )
        width_axis = np.array([-normal[1], normal[0]])
        passage = None
        for sample_index in candidates[candidates >= cursor]:
            alpha = -signed[sample_index] / (signed[sample_index + 1] - signed[sample_index])
            intersection = points[sample_index] + alpha * (points[sample_index + 1] - points[sample_index])
            lateral_error = float((intersection[:2] - center[:2]) @ width_axis)
            vertical_error = float(intersection[2] - center[2])
            if abs(lateral_error) <= GATE_WIDTH_M / 2 and abs(vertical_error) <= GATE_HEIGHT_M / 2:
                passage_time = float(times[sample_index] + alpha * (times[sample_index + 1] - times[sample_index]))
                direction = "forward" if signed[sample_index + 1] > signed[sample_index] else "reverse"
                passage = (sample_index, passage_time, intersection, lateral_error, vertical_error, direction)
                cursor = sample_index + 1
                break

        row: dict[str, float | int | str | bool] = {
            "dataset": dataset_name,
            "gate_index": gate_index,
            "gate": label,
            "gate_x_m": float(center[0]),
            "gate_y_m": float(center[1]),
            "gate_z_ned_m": float(center[2]),
            "gate_width_m": GATE_WIDTH_M,
            "gate_height_m": GATE_HEIGHT_M,
            "passed": passage is not None,
            "passage_time_s": float("nan"),
            "passage_x_m": float("nan"),
            "passage_y_m": float("nan"),
            "passage_z_ned_m": float("nan"),
            "lateral_error_m": float("nan"),
            "vertical_error_m": float("nan"),
            "crossing_direction": "",
        }
        if passage is not None:
            _, passage_time, intersection, lateral_error, vertical_error, direction = passage
            row.update(
                {
                    "passage_time_s": passage_time,
                    "passage_x_m": float(intersection[0]),
                    "passage_y_m": float(intersection[1]),
                    "passage_z_ned_m": float(intersection[2]),
                    "lateral_error_m": lateral_error,
                    "vertical_error_m": vertical_error,
                    "crossing_direction": direction,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _gate_groups(gates: np.ndarray, passages: pd.DataFrame) -> list[tuple[np.ndarray, list[int], str]]:
    groups: dict[tuple[float, float, float], list[int]] = defaultdict(list)
    for idx, center in enumerate(gates):
        groups[tuple(np.round(center, 6))].append(idx)
    output = []
    for key, indices in groups.items():
        statuses = passages.iloc[indices]["passed"].astype(bool).to_numpy()
        color = "#009E73" if statuses.all() else ("#D55E00" if not statuses.any() else "#E69F00")
        output.append((np.asarray(key, dtype=float), indices, color))
    return output


def _draw_gate_frames(
    ax, gates: np.ndarray, labels: list[str], passages: pd.DataFrame, normals: np.ndarray | None = None
) -> None:
    if normals is None:
        normals = gate_normals_xy(gates)
    for center, indices, color in _gate_groups(gates, passages):
        normal = normals[indices[0]]
        width_axis = np.array([-normal[1], normal[0], 0.0])
        vertical_axis = np.array([0.0, 0.0, 1.0])
        corners = [
            center - width_axis * GATE_WIDTH_M / 2 - vertical_axis * GATE_HEIGHT_M / 2,
            center + width_axis * GATE_WIDTH_M / 2 - vertical_axis * GATE_HEIGHT_M / 2,
            center + width_axis * GATE_WIDTH_M / 2 + vertical_axis * GATE_HEIGHT_M / 2,
            center - width_axis * GATE_WIDTH_M / 2 + vertical_axis * GATE_HEIGHT_M / 2,
        ]
        corners.append(corners[0])
        corners_array = np.asarray(corners)
        ax.plot(corners_array[:, 0], corners_array[:, 1], corners_array[:, 2], color=color, lw=2.4, zorder=5)
        annotations = []
        for idx in indices:
            mark = "✓" if bool(passages.iloc[idx]["passed"]) else "✗"
            annotations.append(f"G{idx + 1:02d}{mark}")
        ax.text(
            center[0],
            center[1],
            center[2] - GATE_HEIGHT_M * 0.62,
            "/".join(annotations),
            color=color,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=6,
        )


def plot_trajectory_gate_3d(
    datasets: list[tuple[str, str, pd.DataFrame]],
    passages: list[pd.DataFrame],
    gates: np.ndarray | list[list[float]],
    labels: list[str],
    output_path: Path,
    max_points: int,
    title: str,
    anomaly_intervals: list[dict | None] | None = None,
) -> Path:
    from anomaly_annotations import finish_annotated_layout, mark_trajectory_interval, save_figure
    centers = normalize_gates(gates)
    gate_labels = normalize_gate_labels(labels, len(centers))
    normals = gate_normals(gates)
    figure = plt.figure(figsize=(11.5, 8.5) if len(datasets) == 1 else (16.0, 7.4), constrained_layout=True)
    axes = [figure.add_subplot(1, len(datasets), idx + 1, projection="3d") for idx in range(len(datasets))]
    all_points = [item[2][["pos_x_m", "pos_y_m", "pos_z_m"]].to_numpy(dtype=float) for item in datasets]
    merged = np.vstack(all_points + [centers])
    limits = []
    for axis_index in range(3):
        values = merged[:, axis_index]
        values = values[np.isfinite(values)]
        lower, upper = float(np.min(values)), float(np.max(values))
        margin = max((upper - lower) * 0.08, 0.25)
        limits.append((lower - margin, upper + margin))

    for idx, (ax, (name, color, df), passage_table) in enumerate(zip(axes, datasets, passages)):
        if len(df) > max_points:
            indices = np.unique(np.linspace(0, len(df) - 1, max_points).astype(int))
            plotted = df.iloc[indices]
        else:
            plotted = df
        ax.plot(plotted["pos_x_m"], plotted["pos_y_m"], plotted["pos_z_m"], color=color, lw=1.15, alpha=0.9)
        ax.scatter(
            plotted["pos_x_m"].iloc[0], plotted["pos_y_m"].iloc[0], plotted["pos_z_m"].iloc[0],
            color="#0072B2", marker="o", s=46, edgecolor="white", linewidth=0.7, zorder=7,
        )
        ax.scatter(
            plotted["pos_x_m"].iloc[-1], plotted["pos_y_m"].iloc[-1], plotted["pos_z_m"].iloc[-1],
            color="#CC79A7", marker="X", s=58, edgecolor="white", linewidth=0.7, zorder=7,
        )
        passed = passage_table[passage_table["passed"].astype(bool)]
        if not passed.empty:
            ax.scatter(
                passed["passage_x_m"], passed["passage_y_m"], passed["passage_z_ned_m"],
                color="#009E73", marker="o", s=22, edgecolor="white", linewidth=0.5, zorder=8,
            )
        _draw_gate_frames(ax, centers, gate_labels, passage_table, normals)
        pass_count = int(passage_table["passed"].astype(bool).sum())
        ax.set_title(f"{name}\n过门：{pass_count}/{len(centers)}", fontsize=12, fontweight="bold")
        ax.set_xlabel("X (m)", labelpad=8)
        ax.set_ylabel("Y (m)", labelpad=8)
        ax.set_zlabel("Z_NED (m)", labelpad=8)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.invert_zaxis()
        ax.set_box_aspect([limits[0][1] - limits[0][0], limits[1][1] - limits[1][0], limits[2][1] - limits[2][0]])
        ax.view_init(elev=27, azim=-58)
        ax.grid(True, alpha=0.25)
        if anomaly_intervals and anomaly_intervals[idx]:
            mark_trajectory_interval(ax, df, anomaly_intervals[idx], ("pos_x_m", "pos_y_m", "pos_z_m"))
            ax.legend(loc="upper left", fontsize=8)

    legend = [
        Line2D([0], [0], color=datasets[0][1], lw=1.5, label="飞行轨迹"),
        Line2D([0], [0], color="#009E73", lw=2.5, label="通过门框"),
        Line2D([0], [0], color="#D55E00", lw=2.5, label="未通过门框"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#0072B2", label="起点"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#CC79A7", label="终点"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=9)
    figure.suptitle(title, fontsize=15, fontweight="bold")
    interval = next((item for item in (anomaly_intervals or []) if item), None)
    if interval:
        finish_annotated_layout(figure, interval)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, output_path, dpi=220, bbox_inches="tight", bbox_extra_artists=[figure._suptitle])
    plt.close(figure)
    return output_path
