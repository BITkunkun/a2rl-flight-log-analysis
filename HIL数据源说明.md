# HIL 数据源

HIL 分析支持两种 CSV，默认使用电脑仿真数据；可在命令行选择，也可在配置顶层写 `"hil_source"`。

| 选项 | 位置 / 速度 / 姿态 | 适用文件 |
| --- | --- | --- |
| `computer_truth` | `pos` / `vel` / `quat` | 电脑仿真 truth CSV（默认） |
| `blackbox_ext` | `extPos` / `extVel` / `extAtt` | Blackbox 提取 CSV；姿态按 roll/pitch/yaw 读取，角度单位由数据判断 |

示例：

```bash
python scripts/analyze_hil_log.py --config configs/hil_sitl_kidney.json --hil-source blackbox_ext
python scripts/compare_hil_sitl.py --config configs/hil_sitl_kidney.json --hil-source computer_truth
```

Blackbox 模式缺少任一必需 `ext*` 字段会报错，不会回退到普通字段。单 HIL、HIL/SITL 和 HIL/SITL/实飞入口均支持此选项；命令行参数优先于配置值。
