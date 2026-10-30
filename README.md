# A2RL 飞行日志分析

本项目用于分析无人机软件在环（SITL）、硬件在环（HIL）、实际飞行和无桨手持实验的 CSV 日志。脚本生成轨迹、速度、高度、姿态、电机和赛道过门图，以及 CSV 指标表和 JSON 摘要，便于检查单次实验和比较不同实验。

本仓库不包含实验日志或报告。配置中的 CSV 路径是示例，使用时请改为自己的文件；计算结果默认写入本地 `reports/`。

## 功能入口

| 场景 | 脚本 | 主要结果 |
| --- | --- | --- |
| SITL 单独分析 | `scripts/analyze_sitl_log.py` | 轨迹、速度、高度、航向、过门、位置设定值差异和电机 |
| HIL 单独分析 | `scripts/analyze_hil_log.py` | 仿真轨迹、速度、姿态、电机与过门；绘图最多前 25 秒 |
| 手持单独分析 | `scripts/analyze_hand_log.py` | 无桨手持轨迹、位置、速度、姿态、角速度和电机 |
| 手持与 SITL | `scripts/compare_hand_sitl.py` | 双方轨迹和控制量对比、加桨趋势和电机指令轨迹的粗略预测 |
| 实飞与 SITL | `scripts/compare_real_sitl_logs.py` | 实飞模式分段、轨迹、姿态、设定值差异、电机和过门 |
| 仿真与实飞通用对比 | `scripts/compare_hil_sitl.py` | HIL、SITL、实飞任意两种或三种；图使用同组最短日志时长 |
| 实飞控制段三方对比 | `scripts/compare_hil_sitl_real.py` | HIL、SITL、实飞，以连续实飞控制段时长为基准绘图 |

`hil_analysis_common.py`、`track_3d.py` 和 `anomaly_annotations.py` 提供共享加载、赛道和绘图功能；`analyze_hand_log.py` 的加载器也供手持对比使用。

## 安装

需要 Python 3.10 或更新版本。进入仓库根目录后：

```bash
python -m pip install -r requirements.txt
```

将自己的 CSV 放在本地，例如 `data/`，或在配置中填写文件绝对路径。`data/`、CSV、`reports/` 和缓存默认不会被 Git 跟踪。

## 数据与坐标约定

| 数据 | 时间 | 位置和速度 | 姿态及其他字段 |
| --- | --- | --- | --- |
| SITL 单独分析 | `timeS` 或 `timeUs` | 优先 `extPos[0..2]` / `extVel[0..2]`，分别回退到 `pos/vel`、`ekf_pos/ekf_vel` | `quat[0..3]`，或 `extAtt` / `ekf_att`；可选设定值、电机和转速 |
| HIL truth | `timeS` 或 `timeUs` | `pos[0..2]` / `vel[0..2]` | `quat[0..3]`；可选 `command[0..3]`、`omega[0..3]` |
| 通用对比中的 SITL | `timeS` 或 `timeUs` | `extPos[0..2]` / `extVel[0..2]` | `quat[0..3]`；可选 `motor[0..3]`、`omega[0..3]` |
| 通用对比中的实飞 | `timeS` 或 `timeUs` | `pos[0..2]` / `vel[0..2]` | `quat[0..3]`；控制段三方对比还需要 `flightModeFlags` |
| 无桨手持 | `timeS` 或 `timeUs` | 优先 `extPos/extVel`，回退 `ekf_pos/ekf_vel`、`pos/vel` | `quat` 或 `extAtt/ekf_att`；可选陀螺仪、电机和 `omegaUnfiltered` |

实飞/SITL 专项对比的实飞侧优先 `pos/vel`，SITL 侧优先 `extPos/extVel`，允许按脚本中的顺序回退。手持对比的电机预测需要相应四路电机字段。

赛道使用 NED 坐标，Z 向下；坐标单位为米，时间为秒，四元数按 `[w, x, y, z]` 解释。请在使用前确认自己的日志和赛道使用相同的坐标约定。HIL 使用电脑仿真的 truth CSV，不需要 Blackbox。

## 配置和运行

### 1. 单 SITL

```bash
python scripts/analyze_sitl_log.py --log data/sitl.csv --label sitl_run --track-config configs/sitl_kidney_track.json
```

`--log` 可以是绝对路径，`--start-s` 和 `--end-s` 可以选择时间段。赛道示例 `configs/sitl_kidney_track.json` 包含 `start` 和 `gates`：每项是 `[x, y, z(NED), yaw(度)]`，yaw 为门平面法向。更换赛道时应修改赛道配置；未指定赛道时脚本会使用内置旧四门。

### 2. 单 HIL 和通用对比

编辑 `configs/hil_sitl_kidney.json` 的 `pairs`，填入所需 CSV 和赛道配置：

```json
{
  "hil_csv": "data/hil_truth.csv",
  "sitl_csv": "data/sitl.csv",
  "real_csv": "data/real.csv"
}
```

单 HIL 入口读取其中的 `hil_csv`；对比入口可使用这三个键中的任意两项或三项。`pairs` 可以有多组。

```bash
python scripts/analyze_hil_log.py --config configs/hil_sitl_kidney.json
python scripts/compare_hil_sitl.py --config configs/hil_sitl_kidney.json
```

### 3. 实飞控制段三方对比

编辑 `configs/kidney_real_hil_sitl.json` 中三种 CSV 路径、`track_config` 和 `real_control_flags`：

```bash
python scripts/compare_hil_sitl_real.py --config configs/kidney_real_hil_sitl.json
```

每组需要 HIL、SITL 和实飞三份文件。实飞模式必须匹配为一个连续控制段；HIL 和 SITL 必须足以覆盖该段时长。配置中的 `[12803, 12291]` 仅为示例，实际使用时需确认对应控制器的模式含义。

### 4. 手持分析与手持/SITL 对比

```bash
python scripts/analyze_hand_log.py --log data/hand.csv --label hand_run
python scripts/compare_hand_sitl.py --config configs/hand_sitl_compare_config.json
```

手持对比前编辑配置中的 `hand_log`、`sitl_log`、门点和其他参数。可选历史校准文件默认留空；预测是粗略趋势筛查，不是实测加桨飞行或完整动力学仿真。

### 5. 实飞/SITL 专项对比

编辑 `configs/real_sitl_compare_config.json` 的实飞和 SITL 路径、赛道及控制模式：

```bash
python scripts/compare_real_sitl_logs.py --config configs/real_sitl_compare_config.json
```

也可用 `--real-log` 和 `--sitl-log` 覆盖配置路径。手持对比和实飞/SITL 专项配置中的 `gates` 使用门中心坐标，方向由脚本推断；单 SITL 和通用 HIL 对比的赛道 JSON 使用带 yaw 的四列格式，两者不可直接混用。

## 输出与解释

各入口默认在 `reports/` 下建立结果目录，包含 `figures/`、`tables/` 和摘要或方法说明。常见输出包括 XY/三维轨迹、速度与高度、姿态、四电机指令和转速、过门偏差，以及均值、标准差、分位数等统计。不同输入字段可用时，实际生成的图片会有所不同。

HIL 单独分析的图最多显示前 25 秒；通用对比的图按最短输入时长截取。实飞控制段三方对比使用选定实飞控制段的时长。完整记录统计和绘图窗口统计分别保存；`plot_window_statistics.csv` 和绘图窗口过门表可用于检查图中范围。

各日志以自己的分析段起点计时，不代表物理事件已经同步；不应将同一时间点的差异直接当作同步跟踪误差。默认过门图匹配一组有序门，可能包含反向或跨圈匹配，不能代表全程逐圈成功率。门框默认 1.5 m × 1.5 m，按位置点判定，未扣除机体尺寸。

设定值差异只有在确认设定值确为动态参考后才能作为跟踪误差；电机信号的单位、归一化方式和通道映射也应先核实。实飞 `pos/vel` 是状态估计，不能直接视为无误差真值。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖时间窗口、字段来源、实飞控制段选择和三方对比的基础行为。实验日志和实验结果由使用者自行提供与保存。
