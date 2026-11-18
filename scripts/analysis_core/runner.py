"""Shared CLI and folder report generation for all supported combinations."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .data import DEFAULT_CONFIG, load_config, load_dataset, resolve_path, window_frame
from .metrics import gate_intervals, parameter_table, summarize
from .plots import render_figures
from .track_3d import evaluate_gate_passages


def markdown_table(frame):
    text = frame.copy().fillna("—")
    for column in text:
        text[column] = text[column].map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    rows = ["| " + " | ".join(text.columns) + " |", "| " + " | ".join(["---"] * len(text.columns)) + " |"]
    rows += ["| " + " | ".join(row) + " |" for row in text.itertuples(index=False, name=None)]
    return "\n".join(rows)


def summary_rows(items, key):
    rows = []
    for item in items:
        stats = item[key]
        gate = stats["first_gate_to_last_gate"]
        rows.append({
            "数据": stats["dataset"], "样本数": stats["rows"], "时长 (s)": stats["duration_s"],
            "水平速度均值 (m/s)": stats["speed_xy_mean_mps"], "P95 (m/s)": stats["speed_xy_p95_mps"],
            "最大速度 (m/s)": stats["speed_xy_max_mps"], "平均高度 (m)": stats["height_mean_m"],
            "正向 / 匹配 / 总门": f"{stats['ordered_gates_forward']} / {stats['ordered_gates_passed']} / {stats['ordered_gates_total']}",
            "首门至末门 (s)": gate["elapsed_s"] if gate else None,
        })
    return pd.DataFrame(rows)


def write_report(output, items, duration, figures, track_path):
    names = " / ".join(item["profile"]["dataset"] for item in items)
    comparison = len(items) > 1
    window_note = "各自分析段的最短共同窗口" if comparison else "所选完整分析段"
    lines = [
        f"# {names} 日志分析", "",
        f"本报告使用{window_note}，目标窗口为 0–{duration:.6f} s。表格和 01–11 图使用窗口内原始采样点。", "",
        "## 分析结果", "", markdown_table(summary_rows(items, "window_summary")), "",
    ]
    for item in items:
        stats = item["window_summary"]
        lines += [
            f"- {stats['dataset']} 的高度范围为 {stats['height_min_m']:.3f}–{stats['height_max_m']:.3f} m，"
            f"Roll、Pitch 绝对值 P95 分别为 {stats['roll_abs_p95_deg']:.3f}°、{stats['pitch_abs_p95_deg']:.3f}°。"
        ]
    gate_times = [item["window_summary"]["first_gate_to_last_gate"] for item in items]
    if comparison and all(gate_times):
        times = "，".join(f"{item['profile']['dataset']} {gate['elapsed_s']:.3f} s"
                          for item, gate in zip(items, gate_times))
        lines += ["", f"首组正向有序门的首门至末门耗时为 {times}。"]
    lines += ["", "## 选段与参数来源", "",
              "| 数据 | 原始时钟 | 选择规则 | 相对日志起止 (s) | 原始时钟起止 (s) | 位置 | 速度 | 姿态 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in items:
        p = item["profile"]
        lines.append(
            f"| {p['dataset']} | {p['time_source']} | {p['selection']} | "
            f"{p['selection_start_s']:.6f}–{p['selection_end_s']:.6f} | "
            f"{p['clock_start_s']:.6f}–{p['clock_end_s']:.6f} | "
            f"{', '.join(p['position_columns'])} | {', '.join(p['velocity_columns'])} | "
            f"{', '.join(p['attitude_columns'])} |"
        )
    lines += [
        "", "坐标统一按 NED 解释，位置单位为 m，速度为 m/s，高度为 −z。"
        "extAtt[0..2] 按 roll、pitch、yaw 弧度转为度。HIL 四元数按 w、x、y、z 归一化后转欧拉角。"
        "Yaw 包裹至 [−180°,180°)，连续航向角通过 unwrap 得到。", "",
        "统计按样本计，标准差使用总体标准差。Yaw 的普通均值受角度包裹影响，"
        "连续航向图用于观察累计旋转。图最多抽样 7000 点，表格使用全部窗口样本。", "",
        "时间按各自分析段首样本归零，同一时间可能对应不同赛道位置。"
        "共同窗口以最短所选数据段为准，边界保留已有样本，最后实际样本可略早于目标末端。"
        "位置、速度、姿态并排图各列共用纵轴范围。", "",
        "电机指令和转速保留原值，具体字段见 source_profile.csv 和图例。"
        "HIL motor_1–4 映射到 command_0–3，rpm_1–4 映射到 omega_0–3。"
        "跨日志电机通道映射及转速量纲需结合记录端定义解释。缺失信号记录有效数和缺失数。", "",
        "## 首组有序过门", "",
    ]
    gate_table = pd.concat([item["passages"] for item in items], ignore_index=True)
    gate_columns = ["dataset", "gate_index", "passed", "passage_time_s",
                    "lateral_error_m", "vertical_error_m", "crossing_direction"]
    lines += [
        markdown_table(gate_table[gate_columns]), "",
        "沿赛道配置按门序匹配轨迹与门平面的交点，交点通过相邻记录线性插值得到。"
        "门框宽、高均为 1.5 m，判定使用位置点。结果记录首组有序门，可能跨圈。"
        "竖向偏差采用 NED Z，负值表示高于门中心。首门至末门统计要求全部门正向匹配。", "",
    ]
    if all(item["gate_window"] is not None for item in items):
        local_end = min(item["gate_window"].time_s.iloc[-1] for item in items)
        lines += [
            f"12–14 图按各自首门事件归零，并使用最短门段的 0–{local_end:.6f} s。"
            "gate_window_statistics.csv 使用这些图对应的记录样本。"
            "summary.json 中首门至末门的时间均值对首末交点插值后积分计算。", "",
        ]
    lines += ["## 图表", ""]
    for filename, title in figures:
        lines += [f"### {title}", "", f"![{title}](figures/{filename})", ""]
    lines += [
        "## 各自完整分析段", "", markdown_table(summary_rows(items, "full_summary")), "",
        "full_segment_statistics.csv 和 full_segment_gate_passages.csv 使用各自完整分析段。"
        "parameter_statistics.csv、gate_passages.csv 和 summary_comparison.csv 使用主分析窗口。"
        "processed 中的 CSV 保存各自完整分析段及其 log_time_s，便于复核窗口和原始时间。", "",
        "## 输入与复现", "", f"赛道文件为 `{track_path}`。", "",
    ]
    for item in items:
        lines.append(f"- {item['profile']['dataset']} 输入为 `{item['profile']['source']}`。")
    lines += ["", "运行命令见项目根目录 RUN_ANALYSIS.md。同名输出目录重跑时更新本组结果。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(kinds, config, gates, labels, output):
    items = []
    for kind in kinds:
        frame, profile = load_dataset(kind, config["datasets"][kind], config["control_flags"])
        full_passages = evaluate_gate_passages(frame, gates, labels, kind)
        items.append({"frame": frame, "profile": profile, "full_passages": full_passages,
                      "full_summary": summarize(frame, profile, full_passages)})
    duration = min(item["profile"]["duration_s"] for item in items)
    for item in items:
        frame = window_frame(item["frame"], duration)
        passages = evaluate_gate_passages(frame, gates, labels, item["profile"]["dataset"])
        item.update(window=frame, passages=passages, window_summary=summarize(frame, item["profile"], passages))
        gate = item["window_summary"]["first_gate_to_last_gate"]
        local = None
        if gate:
            local = frame.loc[frame.time_s.between(gate["start_gate_time_s"], gate["end_gate_time_s"])].copy()
            if len(local) >= 2:
                local["time_s"] -= gate["start_gate_time_s"]
            else:
                local = None
        item["gate_window"] = local
    if all(item["gate_window"] is not None for item in items):
        gate_end = min(item["gate_window"].time_s.iloc[-1] for item in items)
        local_frames = [item["gate_window"].loc[item["gate_window"].time_s <= gate_end].copy() for item in items]
        valid = all(len(frame) >= 2 for frame in local_frames)
        for item, local in zip(items, local_frames):
            item["gate_window"] = local if valid else None

    figures, tables, processed = output / "figures", output / "tables", output / "processed"
    for directory in (figures, tables, processed):
        directory.mkdir(parents=True, exist_ok=True)
    # These outputs depend on optional channels or successful gate matches.
    for filename in ("09_motor_commands.png", "10_motor_rotation.png",
                     "12_gate_window.png", "13_gate_window.png", "14_gate_window.png"):
        (figures / filename).unlink(missing_ok=True)
    (tables / "gate_window_statistics.csv").unlink(missing_ok=True)
    rendered = render_figures(items, gates, labels, figures, duration)
    for filename, key in (("parameter_statistics.csv", "window"), ("full_segment_statistics.csv", "frame")):
        pd.concat([parameter_table(item[key], item["profile"]) for item in items], ignore_index=True).to_csv(
            tables / filename, index=False, encoding="utf-8-sig")
    for filename, key in (("gate_passages.csv", "passages"), ("full_segment_gate_passages.csv", "full_passages")):
        pd.concat([item[key] for item in items], ignore_index=True).to_csv(
            tables / filename, index=False, encoding="utf-8-sig")
    if all(item["gate_window"] is not None for item in items):
        pd.concat([parameter_table(item["gate_window"], item["profile"]) for item in items], ignore_index=True).to_csv(
            tables / "gate_window_statistics.csv", index=False, encoding="utf-8-sig")
    summary_rows(items, "window_summary").to_csv(tables / "summary_comparison.csv", index=False, encoding="utf-8-sig")
    gate_intervals(pd.concat([item["passages"] for item in items])).to_csv(
        tables / "gate_intervals.csv", index=False, encoding="utf-8-sig")
    profiles = []
    for item in items:
        profile = item["profile"]
        profiles.append({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                         for key, value in profile.items()})
        item["frame"].to_csv(processed / f"{profile['dataset']}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(profiles).to_csv(tables / "source_profile.csv", index=False, encoding="utf-8-sig")
    result = {
        "track_config": str(resolve_path(config["track_config"])),
        "window_rule": "full selected segment" if len(items) == 1 else "shortest selected segment",
        "analysis_window_s": [0.0, duration], "physical_time_synchronization": False,
        "datasets": {item["profile"]["dataset"]: {
            "profile": item["profile"], "analysis_window": item["window_summary"],
            "full_segment": item["full_summary"],
        } for item in items},
        "figures": [name for name, _ in rendered],
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(output, items, duration, rendered, config["track_config"])
    print(f"{' / '.join(kinds)} -> {output}")
    return result


def main(kinds):
    parser = argparse.ArgumentParser(description=" / ".join(kinds) + " 统一日志分析")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="统一分析配置")
    for kind in kinds:
        parser.add_argument(f"--{kind.lower()}", help=f"{kind} CSV 路径")
    if len(kinds) == 1:
        parser.add_argument("--log", help="单日志 CSV 路径")
    parser.add_argument("--track-config", help="赛道 JSON")
    parser.add_argument("--out-dir", help="报告目录")
    args = parser.parse_args()
    config, gates, labels = load_config(args.config, args.track_config)
    for kind in kinds:
        override = getattr(args, kind.lower())
        if len(kinds) == 1:
            override = args.log or override
        if override:
            config["datasets"][kind]["csv"] = override
    suffix = "_".join(kind.lower() for kind in kinds) + ("_analysis" if len(kinds) == 1 else "_comparison")
    output = resolve_path(args.out_dir) if args.out_dir else resolve_path(config["output_root"]) / suffix
    analyze(kinds, config, gates, labels, output)
