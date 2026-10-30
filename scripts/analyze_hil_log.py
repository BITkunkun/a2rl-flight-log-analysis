"""Analyze HIL simulator truth CSVs listed in configs/hil_sitl_kidney.json."""
from __future__ import annotations

import argparse

from hil_analysis_common import (
    DEFAULT_CONFIG, MAX_PLOT_POINTS, clear_optional_outputs, distributions_plot, gate_errors_plot, gate_window_plots,
    load_config, load_frame, matched_gate_window, motor_channels_plot, motor_plot, parameter_table, passages_for,
    resolve_path, save_json, setup_plots, summarize, time_plot, truncate_for_plot, xy_plot,
)
from track_3d import plot_trajectory_gate_3d


def analyze(path, output_root, gates, labels, track_path):
    frame, profile = load_frame(path, "HIL")
    passages = passages_for(frame, gates, labels, "HIL")
    summary = summarize(frame, profile, passages)
    plot_end = min(25.0, float(frame.time_s.iloc[-1]))
    plot_frame = truncate_for_plot(frame, plot_end)
    plot_passages = passages_for(plot_frame, gates, labels, "HIL")
    summary["track_config"] = str(resolve_path(track_path).resolve())
    summary["plot_window_s"] = [0.0, plot_end]
    summary["plot_last_sample_s"] = float(plot_frame.time_s.iloc[-1])
    out = output_root / path.stem
    figures = out / "figures"
    tables = out / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    clear_optional_outputs(figures, tables, ("05_motor_raw.png",))
    save_json(out / "summary.json", summary)
    frame.to_csv(tables / "normalized_timeseries.csv", index=False, encoding="utf-8-sig")
    passages.to_csv(tables / "gate_passage_3d.csv", index=False, encoding="utf-8-sig")
    plot_passages.to_csv(tables / "plot_gate_passage_3d.csv", index=False, encoding="utf-8-sig")
    parameter_table(frame, profile).to_csv(tables / "parameter_statistics.csv", index=False, encoding="utf-8-sig")
    parameter_table(plot_frame, profile).to_csv(tables / "plot_window_statistics.csv", index=False, encoding="utf-8-sig")
    xy_plot([("HIL", plot_frame)], gates, figures / "01_trajectory_xy.png")
    time_plot([("HIL", plot_frame)], [("speed_xy_mps", "水平速度 (m/s)"), ("height_m", "高度 (m)")],
              figures / "02_speed_height.png", "HIL 绘图窗口速度和高度", x_end=plot_end)
    time_plot([("HIL", plot_frame)], [("roll_deg", "Roll (°)"), ("pitch_deg", "Pitch (°)"), ("yaw_deg", "Yaw (°)")],
              figures / "03_attitude.png", "HIL 绘图窗口姿态", x_end=plot_end)
    plot_trajectory_gate_3d([("HIL", "#0072B2", plot_frame)], [plot_passages], gates, labels,
                            figures / "04_trajectory_3d_gates.png", MAX_PLOT_POINTS,
                            "HIL 三维轨迹与第一组有序门")
    motor_plot(plot_frame, figures / "05_motor_raw.png", plot_end)
    time_plot([("HIL", plot_frame)], [("pos_x_m", "X (m)"), ("pos_y_m", "Y (m)"), ("pos_z_m", "Z NED (m)")],
              figures / "06_position_components.png", "HIL 位置分量", x_end=plot_end)
    time_plot([("HIL", plot_frame)], [("vel_x_mps", "Vx (m/s)"), ("vel_y_mps", "Vy (m/s)"), ("vel_z_mps", "Vz NED (m/s)")],
              figures / "07_velocity_components.png", "HIL 速度分量", x_end=plot_end)
    motor_channels_plot([("HIL", plot_frame)], "command", figures / "08_motor_commands.png", x_end=plot_end)
    motor_channels_plot([("HIL", plot_frame)], "omega", figures / "09_motor_omega.png", x_end=plot_end)
    distributions_plot([("HIL", plot_frame)], figures / "10_distributions.png")
    gate_errors_plot([("HIL", plot_passages)], figures / "11_gate_errors.png")
    window = matched_gate_window(plot_frame, plot_passages)
    if window is not None:
        parameter_table(window, profile).to_csv(tables / "gate_window_statistics.csv", index=False, encoding="utf-8-sig")
        gate_window_plots([("HIL", window)], figures)
    (out / "分析说明.md").write_text(
        f"# HIL 分析：{path.name}\n\n"
        f"输入：`{path.resolve()}`；赛道：`{resolve_path(track_path).resolve()}`。\n\n"
        f"整段时长 {summary['duration_s']:.3f} s；水平速度均值 {summary['speed_xy_mean_mps']:.3f} m/s；"
        f"第一组有序门匹配 {summary['ordered_gates_passed']}/{summary['ordered_gates_total']}，"
        f"其中正向 {summary['ordered_gates_forward']}。\n\n"
        "坐标假定为 NED、四元数顺序假定为 w,x,y,z。门框为 1.5 m × 1.5 m，按位置点计算，"
        "不考虑机体尺寸；脚本只匹配一组有序门，允许反向，不能作为全程逐圈成功率。"
        "如门未匹配，后续门仍可能被匹配，不能把该序列视为完整圈。\n\n"
        "速度、高度等是整段统计，包含起飞；电机数据按原值展示，单位和通道映射待确认。\n\n"
        "`tables/parameter_statistics.csv` 包含位置、速度、高度、姿态及每路电机信号的样本均值、"
        "标准差、最小值、P05、中位数、P95、最大值及绝对值 P95；四路完整时还包含通道均值和极差。"
        "统计使用全量样本，时间图可能抽样；均值按样本计，航向角均值是包裹角度的普通均值，不是圆周均值。\n\n"
        f"所有图仅使用日志起点至 {plot_end:.3f} s 的数据，最后实际样本为 {plot_frame.time_s.iloc[-1]:.6f} s；"
        "`plot_window_statistics.csv` 与 `plot_gate_passage_3d.csv` 对应该绘图窗口。"
        "`summary.json`、`parameter_statistics.csv` 和 `gate_passage_3d.csv` 仍是全程统计。\n\n"
        "全部门均正向匹配时，`gate_window_statistics.csv` 和 12–16 图显示首门到末门之间的记录样本；"
        "图按首门事件计时，不插值边界。门序仍可能跨圈，不能保证是连续完整圈。\n",
        encoding="utf-8")
    print(f"HIL: {path.name} -> {out}")


def main():
    parser = argparse.ArgumentParser(description="按配置批量分析 HIL truth CSV")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置 JSON；路径相对工作区根目录")
    args = parser.parse_args()
    config, gates, labels = load_config(args.config)
    setup_plots()
    seen = set()
    for pair in config["pairs"]:
        if pair.get("hil_csv"):
            path = resolve_path(pair["hil_csv"])
            if path.resolve() not in seen:
                analyze(path, resolve_path(config["hil_output_root"]), gates, labels, config["track_config"])
                seen.add(path.resolve())


if __name__ == "__main__":
    main()
