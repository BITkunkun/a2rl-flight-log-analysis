"""Statistics with one definition for standalone and comparison reports."""
from __future__ import annotations

import numpy as np
import pandas as pd


def parameter_table(frame, profile):
    rows = []
    values = {col: frame[col] for col in profile["signal_metadata"] if col in frame}
    metadata = dict(profile["signal_metadata"])
    for prefix in ("command", "omega"):
        columns = [f"{prefix}_{i}" for i in range(4)]
        if all(col in frame for col in columns):
            four = frame[columns].where(np.isfinite(frame[columns]))
            complete = four.notna().all(axis=1)
            for suffix, series in (("mean", four.mean(axis=1)), ("spread", four.max(axis=1) - four.min(axis=1))):
                column = f"{prefix}_{suffix}"
                values[column] = series.where(complete)
                metadata[column] = (f"{prefix} four-channel {suffix}", "raw")
    for column, series in values.items():
        finite = series.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        stats = {key: None for key in ("mean", "std", "min", "p05", "median", "p95", "max", "abs_p95")}
        if len(finite):
            stats = {
                "mean": float(np.mean(finite)), "std": float(np.std(finite)),
                "min": float(finite.min()), "p05": float(np.percentile(finite, 5)),
                "median": float(np.median(finite)), "p95": float(np.percentile(finite, 95)),
                "max": float(finite.max()), "abs_p95": float(np.percentile(np.abs(finite), 95)),
            }
        source, unit = metadata[column]
        rows.append({"dataset": profile["dataset"], "signal": column, "source": source, "unit": unit,
                     "valid_count": len(finite), "missing_count": len(frame) - len(finite), **stats})
    return pd.DataFrame(rows)


def gate_segment_metrics(frame, passages):
    if (len(passages) < 2 or not passages.passed.all()
            or not passages.crossing_direction.eq("forward").all()):
        return None
    start, end = passages.passage_time_s.iloc[[0, -1]].to_numpy(dtype=float)
    if end <= start:
        return None
    time = frame.time_s.to_numpy()
    grid = np.r_[start, time[(time > start) & (time < end)], end]
    duration = end - start
    result = {"start_gate_time_s": float(start), "end_gate_time_s": float(end), "elapsed_s": float(duration)}
    for column in ("speed_xy_mps", "height_m"):
        values = np.interp(grid, time, frame[column].to_numpy())
        result[f"{column}_time_mean"] = float(np.trapezoid(values, grid) / duration)
    return result


def summarize(frame, profile, passages):
    points = frame[[f"pos_{a}_m" for a in "xyz"]].to_numpy()
    delta = np.diff(points, axis=0)
    passed = passages.loc[passages.passed]
    return {
        "dataset": profile["dataset"], "rows": len(frame),
        "duration_s": float(frame.time_s.iloc[-1]),
        "path_length_xy_m": float(np.linalg.norm(delta[:, :2], axis=1).sum()),
        "path_length_3d_m": float(np.linalg.norm(delta, axis=1).sum()),
        "speed_xy_mean_mps": float(frame.speed_xy_mps.mean()),
        "speed_xy_p95_mps": float(frame.speed_xy_mps.quantile(.95)),
        "speed_xy_max_mps": float(frame.speed_xy_mps.max()),
        "height_mean_m": float(frame.height_m.mean()), "height_min_m": float(frame.height_m.min()),
        "height_max_m": float(frame.height_m.max()),
        "roll_abs_p95_deg": float(frame.roll_deg.abs().quantile(.95)),
        "pitch_abs_p95_deg": float(frame.pitch_deg.abs().quantile(.95)),
        "yaw_rotation_deg": float(frame.yaw_unwrapped_deg.iloc[-1] - frame.yaw_unwrapped_deg.iloc[0]),
        "ordered_gates_passed": int(passages.passed.sum()),
        "ordered_gates_forward": int(passages.crossing_direction.eq("forward").sum()),
        "ordered_gates_total": len(passages),
        "gate_lateral_abs_max_m": float(passed.lateral_error_m.abs().max()) if len(passed) else None,
        "gate_vertical_abs_max_m": float(passed.vertical_error_m.abs().max()) if len(passed) else None,
        "first_gate_to_last_gate": gate_segment_metrics(frame, passages),
    }


def gate_intervals(passages):
    rows = []
    for _, data in passages.groupby("dataset", sort=False):
        for index in range(1, len(data)):
            before, after = data.iloc[index - 1], data.iloc[index]
            valid = before.passed and after.passed and before.crossing_direction == after.crossing_direction == "forward"
            rows.append({"dataset": before.dataset, "from_gate": int(before.gate_index),
                         "to_gate": int(after.gate_index),
                         "elapsed_s": float(after.passage_time_s - before.passage_time_s) if valid else None})
    return pd.DataFrame(rows, columns=["dataset", "from_gate", "to_gate", "elapsed_s"])
