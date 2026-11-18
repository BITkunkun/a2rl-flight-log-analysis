# A2RL 飞行日志统一分析

统一分析 INDIflight 的 SITL、HIL 和实际飞行 CSV。七个入口共用 scripts/analysis_core，统一字段来源、选段、统计、过门判定和绘图。

| 分析 | 入口 |
| --- | --- |
| 单 SITL | scripts/analyze_sitl_log.py |
| 单 HIL | scripts/analyze_hil_log.py |
| 单 REAL | scripts/analyze_real_log.py |
| SITL / HIL | scripts/compare_sitl_hil.py |
| SITL / REAL | scripts/compare_sitl_real.py |
| HIL / REAL | scripts/compare_hil_real.py |
| SITL / HIL / REAL | scripts/compare_sitl_hil_real.py |

## 安装与运行

需要 Python 3.10 或更高版本。

```powershell
python -m pip install -r requirements.txt
python scripts/analyze_sitl_log.py
python scripts/analyze_hil_log.py
python scripts/analyze_real_log.py
python scripts/compare_sitl_hil.py
python scripts/compare_sitl_real.py
python scripts/compare_hil_real.py
python scripts/compare_sitl_hil_real.py
```

默认配置为 configs/analysis_config.json。使用前，将 CSV 放到配置指定的 data/raw 路径，或修改配置中的路径。原始日志和生成报告由使用者提供及保存在本地。

当前配置对应 Kidney speed6 1028 实验。三个 CSV 路径分别为 data/raw/sitl/Kidney_1023/M23_kidney_speed6_1028.csv、data/raw/hil/M23_kidney_speed6_1028_HIL.csv、data/raw/real/Real_Kidney_speed6_1028.csv。

也可以直接指定项目目录之外的日志。相对路径以本仓库根目录为准，支持绝对路径。

```powershell
python scripts/compare_sitl_hil_real.py --sitl ../data/raw/sitl/Kidney_1023/M23_kidney_speed6_1028.csv --hil ../data/raw/hil/M23_kidney_speed6_1028_HIL.csv --real ../data/raw/real/Real_Kidney_speed6_1028.csv
```

上述命令适用于本仓库位于 A2RL_LOG_analys/a2rl-flight-log-analysis，原始数据位于同级 ../data/raw 的本地布局。

| 参数 | 用途 |
| --- | --- |
| --config | 指定统一配置 JSON |
| --sitl、--hil、--real | 覆盖本入口涉及的 CSV 路径 |
| --log | 单独分析入口的 CSV 路径 |
| --track-config | 覆盖赛道 JSON |
| --out-dir | 指定本次报告文件夹 |

## 字段与选段

| 数据 | 位置 | 速度 | 姿态 | 选段 |
| --- | --- | --- | --- | --- |
| SITL | extPos[0..2] | extVel[0..2] | extAtt[0..2]，roll/pitch/yaw，rad | 完整日志 |
| REAL | extPos[0..2] | extVel[0..2] | extAtt[0..2]，roll/pitch/yaw，rad | 配置 control_flags 的连续控制段 |
| 当前 HIL | x_ned/y_ned/z_ned | vx_ned/vy_ned/vz_ned | qw/qx/qy/qz，w/x/y/z | nn_mode=1 连续段 |

SITL、REAL 必须提供完整的 extPos/extVel/extAtt。导入的 Blackbox CSV 应已完成单位换算，位置为 m，速度为 m/s，姿态为 rad。HIL 四元数转换前归一化，输出姿态为 deg。坐标按 NED 解释，高度取 −z。

REAL 默认控制标记为 12803、12291，需按实验实际模式配置。存在多个不连续控制段时，程序要求提供本次实验的连续日志。

旧电脑仿真 HIL 支持 pos[0..2]、vel[0..2]、quat[0..3]。缺少模式标记时，需要在 datasets.HIL 中明确 nn_start_s，单位为相对原日志起点的秒数。HIL Blackbox 支持 extPos/extVel/extAtt，并按 flightModeFlags 选段。

数据来源的上游代码核对、单位依据与当前实验的数据质量说明见 [DATA_SOURCES.md](DATA_SOURCES.md)。

## 报告文件夹

默认输出到 reports/kidney_speed6_1028_unified 下的七个子目录。

| 文件或目录 | 内容 |
| --- | --- |
| report.md | 中文指标、来源、分析口径、图表 |
| summary.json | 采样、选段、共同窗口与各自完整段的指标 |
| figures/ | 轨迹、三维门框、位置、速度、高度、姿态、连续航向、过门偏差、电机信号、分布和首门局部图 |
| tables/parameter_statistics.csv | 主窗口信号统计 |
| tables/gate_passages.csv | 主窗口首组有序过门时间、交点和偏差 |
| tables/gate_intervals.csv | 相邻正向门耗时 |
| tables/summary_comparison.csv | 主窗口指标汇总 |
| tables/source_profile.csv | 原始字段、路径、采样和选段来源 |
| tables/full_segment_statistics.csv | 各自完整分析段的信号统计 |
| tables/full_segment_gate_passages.csv | 各自完整分析段首组有序过门 |
| tables/gate_window_statistics.csv | 首门局部图对应的样本统计，全部门正向匹配时生成 |
| processed/ | 本入口涉及数据的完整选段，保留原日志相对时间 log_time_s |

单独分析覆盖完整选段。组合分析取最短选段时长，按各自首样本归零。主图和主表使用相同窗口。相同时间可能对应不同赛道位置。

信号统计包括有效数、缺失数、均值、总体标准差、最小值、P05、中位数、P95、最大值、绝对值 P95。四路电机同时有效时计算通道均值和极差。电机单位保留记录端原值。

赛道 gates 使用 [x,y,z_NED,yaw_度]，yaw 为门平面法向。门框宽高各 1.5 m，交点通过相邻记录线性插值。过门结果为首组有序门，可能跨圈，判定使用位置点。全部门正向匹配时，局部图按首门归零，再取最短门段，时间均值使用交点边界积分。

同名重跑更新对应报告目录。旧版本可通过 Git 历史查询。
