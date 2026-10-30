"""Compare any two or three independent HIL, SITL and flight recordings."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from hil_analysis_common import (
    DEFAULT_CONFIG, MAX_PLOT_POINTS, clear_optional_outputs, distributions_plot, gate_errors_plot, gate_window_plots,
    load_config, load_frame, matched_gate_window, motor_channels_plot, parameter_table,
    passages_for, resolve_path, save_json, select_real_control_segment, setup_plots, summarize, time_plot, truncate_for_plot, xy_plot,
)
from track_3d import plot_trajectory_gate_3d

KINDS = (("hil_csv", "HIL", "#0072B2"), ("sitl_csv", "SITL", "#D55E00"), ("real_csv", "REAL", "#009E73"))


def gate_segment(frame, gates):
    """Use the matched first and final gate only when all matches are forward."""
    if not gates.passed.all() or not (gates.crossing_direction == "forward").all():
        return None, None
    start, end = gates.passage_time_s.iloc[[0, -1]].to_numpy(dtype=float)
    if end <= start:
        return None, None
    t = frame.time_s.to_numpy()
    grid = np.r_[start, t[(t > start) & (t < end)], end]
    segment = pd.DataFrame({"time_s": grid})
    for col in ("pos_x_m", "pos_y_m", "pos_z_m", "speed_xy_mps", "height_m"):
        segment[col] = np.interp(grid, t, frame[col].to_numpy())
    duration = end - start

    def time_mean(column):
        values = segment[column].to_numpy()
        return float(np.sum((values[:-1] + values[1:]) * np.diff(grid) / 2) / duration)

    return segment, {"start_gate_time_s": float(start), "end_gate_time_s": float(end),
                     "elapsed_s": float(duration), "speed_xy_time_mean_mps": time_mean("speed_xy_mps"),
                     "height_time_mean_m": time_mean("height_m")}


def selected_inputs(pair):
    """The existing hil_csv/sitl_csv configuration also accepts real_csv."""
    chosen = [(kind, color, resolve_path(pair[key])) for key, kind, color in KINDS if pair.get(key)]
    if len(chosen) < 2:
        raise ValueError("每组对比至少需要 hil_csv、sitl_csv、real_csv 中的两个路径")
    return chosen


def compare(pair, output_root, gates, labels, track_path, *, real_control_flags=None, reference_kind=None):
    chosen = selected_inputs(pair)
    datasets = []
    for kind, color, path in chosen:
        frame, profile = load_frame(path, kind)
        if kind == "REAL" and real_control_flags is not None:
            frame, profile = select_real_control_segment(frame, profile, real_control_flags)
        passage = passages_for(frame, gates, labels, kind)
        _, segment_metrics = gate_segment(frame, passage)
        datasets.append({"kind": kind, "color": color, "path": path, "frame": frame,
                         "profile": profile, "passage": passage, "summary": summarize(frame, profile, passage),
                         "segment": segment_metrics})
    if reference_kind:
        reference = next((d for d in datasets if d["kind"] == reference_kind), None)
        if reference is None:
            raise ValueError(f"对比缺少参考数据 {reference_kind}")
        plot_end = float(reference["frame"].time_s.iloc[-1])
        short = [d["kind"] for d in datasets if d["frame"].time_s.iloc[-1] + 1e-9 < plot_end]
        if short:
            raise ValueError(f"{', '.join(short)} 记录短于 {reference_kind} 参考段，无法覆盖 {plot_end:.3f} s")
    else:
        plot_end = min(float(item["frame"].time_s.iloc[-1]) for item in datasets)
    for item in datasets:
        item["plot_frame"] = truncate_for_plot(item["frame"], plot_end)
        item["plot_passage"] = passages_for(item["plot_frame"], gates, labels, item["kind"])
        item["plot_segment"], _ = gate_segment(item["plot_frame"], item["plot_passage"])
        item["plot_window"] = matched_gate_window(item["plot_frame"], item["plot_passage"])

    out = output_root / "_vs_".join(item["path"].stem for item in datasets)
    figures, tables = out / "figures", out / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    clear_optional_outputs(figures, tables, ("05_gate_segment_xy.png",))
    summary = {
        "track_config": str(resolve_path(track_path).resolve()),
        "comparison": "independent runs; plot window starts at each selected analysis segment; no physical synchronization",
        "plot_window_s": [0.0, plot_end],
        "plot_reference": reference_kind or "shortest",
        "plot_last_sample_s": {d["kind"]: float(d["plot_frame"].time_s.iloc[-1]) for d in datasets},
        "first_gate_to_last_gate": {d["kind"]: d["segment"] for d in datasets},
    }
    summary.update({d["kind"]: d["summary"] for d in datasets})
    save_json(out / "summary.json", summary)
    pd.DataFrame([d["summary"] for d in datasets]).to_csv(tables / "summary_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat([d["passage"] for d in datasets], ignore_index=True).to_csv(tables / "gate_passages.csv", index=False, encoding="utf-8-sig")
    pd.concat([d["plot_passage"] for d in datasets], ignore_index=True).to_csv(tables / "plot_gate_passages.csv", index=False, encoding="utf-8-sig")
    for filename, frame_key in (("parameter_statistics.csv", "frame"), ("plot_window_statistics.csv", "plot_frame")):
        pd.concat([parameter_table(d[frame_key], d["profile"]) for d in datasets], ignore_index=True).to_csv(
            tables / filename, index=False, encoding="utf-8-sig")
    intervals = []
    for d in datasets:
        passage = d["passage"]
        for i in range(1, len(passage)):
            prev, current = passage.iloc[i - 1], passage.iloc[i]
            valid = prev.passed and current.passed and prev.crossing_direction == current.crossing_direction == "forward"
            intervals.append({"dataset": d["kind"], "from_gate": int(prev.gate_index), "to_gate": int(current.gate_index),
                              "elapsed_s": float(current.passage_time_s - prev.passage_time_s) if valid else None})
    pd.DataFrame(intervals).to_csv(tables / "gate_intervals.csv", index=False, encoding="utf-8-sig")

    plots = [(d["kind"], d["plot_frame"]) for d in datasets]
    xy_plot(plots, gates, figures / "01_trajectory_xy.png")
    time_plot(plots, [("speed_xy_mps", "水平速度 (m/s)"), ("height_m", "高度 (m)")],
              figures / "02_speed_height.png", "共同绘图窗口：速度和高度，时间未同步", x_end=plot_end)
    time_plot(plots, [("roll_deg", "Roll (°)"), ("pitch_deg", "Pitch (°)"), ("yaw_deg", "Yaw (°)")],
              figures / "03_attitude.png", "共同绘图窗口：姿态，时间未同步", x_end=plot_end)
    plot_trajectory_gate_3d([(d["kind"], d["color"], d["plot_frame"]) for d in datasets],
                            [d["plot_passage"] for d in datasets], gates, labels,
                            figures / "04_trajectory_3d_gates.png", MAX_PLOT_POINTS, "绘图窗口内轨迹与有序过门")
    if all(d["plot_segment"] is not None for d in datasets):
        xy_plot([(d["kind"] + " 首门至末门", d["plot_segment"]) for d in datasets], gates, figures / "05_gate_segment_xy.png")
    time_plot(plots, [("pos_x_m", "X (m)"), ("pos_y_m", "Y (m)"), ("pos_z_m", "Z NED (m)")],
              figures / "06_position_components.png", "共同绘图窗口：位置，时间未同步", x_end=plot_end)
    time_plot(plots, [("vel_x_mps", "Vx (m/s)"), ("vel_y_mps", "Vy (m/s)"), ("vel_z_mps", "Vz NED (m/s)")],
              figures / "07_velocity_components.png", "共同绘图窗口：速度，时间未同步", x_end=plot_end)
    motor_channels_plot(plots, "command", figures / "08_motor_commands.png", x_end=plot_end)
    motor_channels_plot(plots, "omega", figures / "09_motor_omega.png", x_end=plot_end)
    distributions_plot(plots, figures / "10_distributions.png")
    gate_errors_plot([(d["kind"], d["plot_passage"]) for d in datasets], figures / "11_gate_errors.png")
    windows = [d for d in datasets if d["plot_window"] is not None]
    if windows:
        pd.concat([parameter_table(d["plot_window"], d["profile"]) for d in windows], ignore_index=True).to_csv(
            tables / "gate_window_statistics.csv", index=False, encoding="utf-8-sig")
    if len(windows) == len(datasets):
        gate_window_plots([(d["kind"], d["plot_window"]) for d in datasets], figures)

    lines = [f"# {' / '.join(d['kind'] for d in datasets)} 对比", ""]
    lines += [f"{d['kind']}：`{d['path'].resolve()}`" for d in datasets]
    window_reason = f"{reference_kind} 参考段时长" if reference_kind else "最短日志时长"
    lines += ["", f"赛道：`{resolve_path(track_path).resolve()}`", "",
              f"绘图统一使用各自所选数据起点后的 0–{plot_end:.3f} s（{window_reason}）；最后一个实际样本可能稍早于边界。",
              "", "| 数据 | 全程时长 (s) | 全程水平速度均值 (m/s) | 全程有序门匹配 | 正向门 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {s['dataset']} | {s['duration_s']:.3f} | {s['speed_xy_mean_mps']:.3f} | "
              f"{s['ordered_gates_passed']}/{s['ordered_gates_total']} | {s['ordered_gates_forward']} |"
              for s in (d["summary"] for d in datasets)]
    real = next((d for d in datasets if d["kind"] == "REAL"), None)
    if real and real["profile"].get("selection") == "flightModeFlags":
        p = real["profile"]
        lines += ["", f"实飞按 flightModeFlags {p['control_flags']} 选取原日志 "
                  f"{p['selection_start_s']:.3f}–{p['selection_end_s']:.3f} s 的连续控制段，"
                  "其统计均针对该段；HIL/SITL 的全程统计仍覆盖各自完整输入。"]
    lines += ["", "`parameter_statistics.csv`、`gate_passages.csv` 和摘要为所选数据全程；"
              "`plot_window_statistics.csv`、`plot_gate_passages.csv` 及所有图使用共同绘图窗口。"
              "首门局部图以各自首门为起点，并按最短门段截取；窗口内记录不插值。",
              "", "日志起点各自归零不表示物理同步；相同时间不一定是相同赛道位置，因此不计算逐点误差。"
              "全部门匹配不保证连续完整圈；门框 1.5 m × 1.5 m，按位置点判定，未考虑机体尺寸。",
              "", "HIL 使用 pos/vel，SITL 使用 extPos/extVel，REAL 使用 pos/vel；假定同一 NED 坐标与 w,x,y,z 四元数。"
              "需核实坐标系和所选飞行区间；电机单位及通道映射未确认，不做数值差分。",
              "", "均值按样本计，航向普通均值不是圆周均值；图可能抽样，统计表使用全量样本。"]
    (out / "分析说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison -> {out}")
    return datasets, plot_end


def main():
    parser = argparse.ArgumentParser(description="按配置对比 HIL、SITL 和实飞中的任意两种或三种 CSV")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置 JSON；路径相对工作区根目录")
    args = parser.parse_args()
    config, gates, labels = load_config(args.config)
    setup_plots()
    root = resolve_path(config["compare_output_root"])
    results, overview = [], []
    for pair in config["pairs"]:
        datasets, plot_end = compare(pair, root, gates, labels, config["track_config"])
        name = " / ".join(d["path"].stem for d in datasets)
        for d in datasets:
            s, segment = d["summary"], d["segment"]
            results.append(s)
            elapsed = f"{segment['elapsed_s']:.3f}" if segment else "未满足条件"
            speed = f"{segment['speed_xy_time_mean_mps']:.3f}" if segment else "—"
            overview.append(f"| {name} | {s['dataset']} | {s['duration_s']:.3f} | {plot_end:.3f} | "
                            f"{s['ordered_gates_passed']}/{s['ordered_gates_total']} (正向 {s['ordered_gates_forward']}) | {elapsed} | {speed} |")
    pd.DataFrame(results).to_csv(root / "all_pairs_summary.csv", index=False, encoding="utf-8-sig")
    (root / "对比汇总.md").write_text(
        "# 仿真 / 实飞对比汇总\n\n| 文件 | 数据 | 全程时长 (s) | 绘图时长 (s) | 全程有序过门 | 首门至末门 (s) | 该段水平速度时间均值 (m/s) |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n" + "\n".join(overview)
        + "\n\n绘图以各自日志起点为零，并截取至同组最短时长；不代表物理同步。全程统计与绘图窗口统计分开保存。"
          "过门序列可能跨圈，不代表完整圈或全程成功率。\n", encoding="utf-8")


if __name__ == "__main__":
    main()
