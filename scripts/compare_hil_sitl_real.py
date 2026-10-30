"""Compare HIL and SITL against the selected REAL control interval."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from compare_hil_sitl import compare, selected_inputs
from hil_analysis_common import ROOT, load_config, resolve_path, setup_plots


def main():
    parser = argparse.ArgumentParser(description="以实飞控制段为时长基准的 HIL/SITL/实飞三方对比")
    parser.add_argument("--config", default=str(ROOT / "configs" / "kidney_real_hil_sitl.json"), help="三方对比配置")
    args = parser.parse_args()
    config, gates, labels = load_config(args.config)
    output_root = resolve_path(config["compare_output_root"])
    flags = config["real_control_flags"]
    setup_plots()
    overview = []
    real_gates = []
    for pair in config["pairs"]:
        if {kind for kind, _, _ in selected_inputs(pair)} != {"HIL", "SITL", "REAL"}:
            raise ValueError("每组必须提供 hil_csv、sitl_csv 和 real_csv")
        datasets, end = compare(pair, output_root, gates, labels, config["track_config"],
                                real_control_flags=flags, reference_kind="REAL")
        name = Path(pair["real_csv"]).stem
        real = next(item for item in datasets if item["kind"] == "REAL")
        passed = real["plot_passage"].loc[real["plot_passage"].passed]
        if not passed.empty:
            largest = passed.loc[passed.lateral_error_m.abs().idxmax()]
            real_gates.append((name, int(largest.gate_index), float(largest.lateral_error_m),
                               float(largest.gate_width_m / 2 - abs(largest.lateral_error_m))))
        for item in datasets:
            frame, passage = item["plot_frame"], item["plot_passage"]
            overview.append({"组别": name, "数据": item["kind"], "参考时长_s": end,
                             "最后采样_s": float(frame.time_s.iloc[-1]),
                             "窗口样本数": len(frame),
                             "水平速度均值_mps": float(frame.speed_xy_mps.mean()),
                             "水平速度P95_mps": float(frame.speed_xy_mps.quantile(.95)),
                             "水平速度最大_mps": float(frame.speed_xy_mps.max()),
                             "平均高度_m": float(frame.height_m.mean()),
                             "有序门匹配": int(passage.passed.sum()),
                             "正向过门": int((passage.crossing_direction == "forward").sum()),
                             "总门数": len(passage)})
    table = pd.DataFrame(overview)
    table.to_csv(output_root / "窗口指标汇总.csv", index=False, encoding="utf-8-sig")
    lines = ["# HIL、SITL 与实飞三方对比", "",
             f"实飞选择 flightModeFlags 为 {flags} 的连续控制段；三方图均取实飞控制段的时长。", "",
             "| 实飞日志 | 数据 | 绘图时长 (s) | 水平速度均值 (m/s) | 水平速度 P95 (m/s) | 平均高度 (m) | 有序门（正向） |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in overview:
        lines.append(f"| {row['组别']} | {row['数据']} | {row['参考时长_s']:.3f} | "
                     f"{row['水平速度均值_mps']:.3f} | {row['水平速度P95_mps']:.3f} | "
                     f"{row['平均高度_m']:.3f} | {row['有序门匹配']}/{row['总门数']} ({row['正向过门']}) |")
    if real_gates:
        lines += ["", "实飞在绘图窗口内匹配到的门中，横向偏差绝对值最大的事件：", ""]
        for name, gate, lateral, point_margin in real_gates:
            lines.append(f"- {name}：第 {gate} 门横向偏差 {lateral:+.3f} m；"
                         f"按门框半宽计算，位置点距侧边约 {point_margin:.3f} m。")
        lines += ["", "上述距离基于实飞位置估计和门位模型，未扣机体尺寸，不能作为实际机体净空。"]
    lines += ["", "实飞从控制段开始计时，HIL 和 SITL 从各自日志起点计时；仅统一展示时长，"
              "不代表三个实验已物理同步，也不计算逐点误差。采样间隔不同，最后实际采样点可能稍早于窗口末端。",
              "", "过门统计仅匹配一组有序门，可能跨圈，不能代表全程逐圈成功率；门框按 1.5 m × 1.5 m "
              "的位置点判定，不考虑机体尺寸。HIL 与 SITL 的全程统计仍保留在各组目录内。",
              "", "位置、速度和姿态按同一 NED 坐标系与四元数约定处理；实飞的 pos/vel 是状态估计。"
              "电机指令与转速按原始值分别展示，单位和跨数据通道映射未经确认。"]
    (output_root / "三方对比汇总.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
