import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_hil_sitl import compare, gate_segment, selected_inputs
from hil_analysis_common import clear_optional_outputs, load_frame, matched_gate_window, parameter_table, select_real_control_segment, setup_plots, truncate_for_plot


class HilAnalysisTests(unittest.TestCase):
    def test_real_control_reference_uses_selected_duration_in_all_plot_frames(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root, pair = Path(tmp), {}
            for name in ("xy_plot", "time_plot", "motor_channels_plot", "distributions_plot", "gate_errors_plot",
                         "gate_window_plots", "plot_trajectory_gate_3d"):
                stack.enter_context(patch(f"compare_hil_sitl.{name}"))
            for kind, count in (("HIL", 7), ("SITL", 6), ("REAL", 8)):
                data = {"timeS": list(range(count))}
                for prefix in (("extPos", "extVel") if kind == "SITL" else ("pos", "vel")):
                    for i in range(3):
                        data[f"{prefix}[{i}]"] = [float(i)] * count
                for i in range(4):
                    data[f"quat[{i}]"] = [1. if i == 0 else 0.] * count
                if kind == "REAL":
                    data["flightModeFlags"] = [0, 4099, 12803, 12291, 12291, 12291, 4099, 4099]
                path = root / f"{kind}.csv"
                pd.DataFrame(data).to_csv(path, index=False)
                pair[f"{kind.lower()}_csv"] = str(path)
            datasets, end = compare(pair, root / "out", np.array([[5., 0., -1., 0.]]), ["Gate 01"], path,
                                    real_control_flags=[12803, 12291], reference_kind="REAL")
            self.assertEqual(end, 3.)
            self.assertTrue(all(d["plot_frame"].time_s.max() == 3. for d in datasets))
            real = next(d for d in datasets if d["kind"] == "REAL")
            self.assertEqual(real["profile"]["selection_start_s"], 2.)
            self.assertEqual(real["profile"]["selection_end_s"], 5.)
            self.assertEqual(real["profile"]["rows"], 4)

    def test_real_control_segment_requires_contiguous_modes_and_preserves_offsets(self):
        frame = pd.DataFrame({"time_s": [0., 1., 2., 3., 4.],
                              "flight_mode_flags": [0, 12803, 12291, 12291, 4099]})
        profile = {"duration_s": 4., "rows": 5}
        selected, details = select_real_control_segment(frame, profile, [12803, 12291])
        self.assertEqual(selected.time_s.tolist(), [0., 1., 2.])
        self.assertEqual(details["selection_start_s"], 1.)
        self.assertEqual(details["selection_end_s"], 3.)
        self.assertEqual(details["full_log_rows"], 5)
        frame.loc[2, "flight_mode_flags"] = 4099
        with self.assertRaisesRegex(ValueError, "不连续"):
            select_real_control_segment(frame, profile, [12803, 12291])
        with self.assertRaisesRegex(ValueError, "缺少"):
            select_real_control_segment(frame.drop(columns="flight_mode_flags"), profile, [12803])

    def test_reference_interval_rejects_shorter_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair = {}
            for kind, count in (("HIL", 3), ("SITL", 4), ("REAL", 5)):
                data = {"timeS": list(range(count))}
                for prefix in (("extPos", "extVel") if kind == "SITL" else ("pos", "vel")):
                    for i in range(3):
                        data[f"{prefix}[{i}]"] = [float(i)] * count
                for i in range(4):
                    data[f"quat[{i}]"] = [1. if i == 0 else 0.] * count
                path = root / f"{kind}.csv"
                pd.DataFrame(data).to_csv(path, index=False)
                pair[f"{kind.lower()}_csv"] = str(path)
            with self.assertRaisesRegex(ValueError, "短于 REAL"):
                compare(pair, root, np.array([[5., 0., -1., 0.]]), ["Gate 01"], path, reference_kind="REAL")

    def test_rerun_removes_only_conditional_generated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            figures, tables = Path(tmp) / "figures", Path(tmp) / "tables"
            figures.mkdir()
            tables.mkdir()
            (figures / "12_gate_window_position.png").write_text("old")
            (tables / "gate_window_statistics.csv").write_text("old")
            (figures / "user_note.txt").write_text("keep")
            clear_optional_outputs(figures, tables)
            self.assertFalse((figures / "12_gate_window_position.png").exists())
            self.assertFalse((tables / "gate_window_statistics.csv").exists())
            self.assertTrue((figures / "user_note.txt").exists())

    def test_three_way_comparison_renders_actual_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair = {}
            for kind, count in (("HIL", 6), ("SITL", 5), ("REAL", 4)):
                path = root / f"{kind}.csv"
                pos, vel = (("extPos", "extVel") if kind == "SITL" else ("pos", "vel"))
                data = {"timeS": list(range(count))}
                for i in range(3):
                    data[f"{pos}[{i}]"] = [float(t) if i == 0 else (-1. if i == 2 else 0.) for t in range(count)]
                    data[f"{vel}[{i}]"] = [1. if i == 0 else 0.] * count
                for i in range(4):
                    data[f"quat[{i}]"] = [1. if i == 0 else 0.] * count
                pd.DataFrame(data).to_csv(path, index=False)
                pair[f"{kind.lower()}_csv"] = str(path)
            setup_plots()
            datasets, end = compare(pair, root / "out", np.array([[1., 0., -1., 0.]]), ["Gate 01"], path)
            out = root / "out" / "HIL_vs_SITL_vs_REAL"
            self.assertEqual(end, 3.)
            self.assertEqual(len(datasets), 3)
            self.assertGreater((out / "figures" / "04_trajectory_3d_gates.png").stat().st_size, 0)
            table = pd.read_csv(out / "tables" / "plot_window_statistics.csv")
            self.assertEqual(set(table.dataset), {"HIL", "SITL", "REAL"})

    def test_truncate_uses_inclusive_recorded_samples_without_interpolation(self):
        frame = pd.DataFrame({"time_s": [0., 10., 20., 25., 26.]})
        self.assertEqual(truncate_for_plot(frame, 25.).time_s.tolist(), [0., 10., 20., 25.])
        self.assertEqual(truncate_for_plot(frame, 20.3).time_s.tolist(), [0., 10., 20.])
        self.assertEqual(truncate_for_plot(frame, 30.).time_s.tolist(), frame.time_s.tolist())

    def test_combinations_use_shortest_time_and_keep_full_statistics(self):
        def write_log(path, kind, count):
            pos = "extPos" if kind == "SITL" else "pos"
            vel = "extVel" if kind == "SITL" else "vel"
            data = {"timeS": list(range(count))}
            for prefix in (pos, vel):
                for i in range(3):
                    data[f"{prefix}[{i}]"] = [float(i)] * count
            for i in range(4):
                data[f"quat[{i}]"] = [1. if i == 0 else 0.] * count
            pd.DataFrame(data).to_csv(path, index=False)

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            for name in ("xy_plot", "time_plot", "motor_channels_plot", "distributions_plot", "gate_errors_plot",
                         "gate_window_plots", "plot_trajectory_gate_3d"):
                stack.enter_context(patch(f"compare_hil_sitl.{name}"))
            import compare_hil_sitl as module
            paths = {kind: root / f"{kind}.csv" for kind in ("HIL", "SITL", "REAL")}
            for kind, count in (("HIL", 6), ("SITL", 4), ("REAL", 3)):
                write_log(paths[kind], kind, count)
            for index, kinds in enumerate((("HIL", "SITL"), ("SITL", "REAL"), ("HIL", "REAL"), ("HIL", "SITL", "REAL"))):
                pair = {f"{kind.lower()}_csv": str(paths[kind]) for kind in kinds}
                self.assertEqual(len(selected_inputs(pair)), len(kinds))
                module.time_plot.reset_mock()
                datasets, end = compare(pair, root / f"out{index}", np.array([[5., 0., -1., 0.]]), ["Gate 01"], paths["HIL"])
                expected = min(len(pd.read_csv(paths[kind])) - 1 for kind in kinds)
                self.assertEqual(end, expected)
                for item in datasets:
                    self.assertLessEqual(item["plot_frame"].time_s.max(), end)
                    self.assertEqual(item["summary"]["rows"], len(pd.read_csv(item["path"])))
                for call in module.time_plot.call_args_list:
                    self.assertEqual(call.kwargs["x_end"], end)
                    self.assertTrue(all(frame.time_s.max() <= end for _, frame in call.args[0]))
            with self.assertRaises(ValueError):
                selected_inputs({"hil_csv": str(paths["HIL"])})

    def test_gate_window_uses_recorded_samples_and_first_gate_origin(self):
        frame = pd.DataFrame({"time_s": [0., 1., 2., 3.], "command_0": [10., 11., 12., 13.]})
        gates = pd.DataFrame({"passed": [True, True], "crossing_direction": ["forward", "forward"],
                              "passage_time_s": [0.5, 2.5]})
        window = matched_gate_window(frame, gates)
        self.assertEqual(window.time_s.tolist(), [0.5, 1.5])
        self.assertEqual(window.command_0.tolist(), [11., 12.])
        gates.loc[1, "passed"] = False
        self.assertIsNone(matched_gate_window(frame, gates))

    def test_motor_statistics_require_complete_finite_channels(self):
        frame = pd.DataFrame({"command_0": [0., 1., 2.], "command_1": [2., 3., 4.],
                              "command_2": [4., float("nan"), 6.], "command_3": [6., 7., 8.]})
        profile = {"dataset": "HIL", "position_source": "pos", "velocity_source": "vel",
                   "command_source": "command", "omega_source": "omega"}
        stats = parameter_table(frame, profile).set_index("signal")
        self.assertEqual(stats.loc["command_2", "valid_count"], 2)
        self.assertEqual(stats.loc["command_mean", "valid_count"], 2)
        self.assertEqual(stats.loc["command_mean", "missing_count"], 1)
        self.assertAlmostEqual(stats.loc["command_mean", "mean"], 4.)
        self.assertAlmostEqual(stats.loc["command_spread", "p95"], 6.)

    def test_segment_uses_interpolated_boundaries_and_time_weighted_mean(self):
        frame = pd.DataFrame({
            "time_s": [0., 1., 3.], "pos_x_m": [0., 1., 3.],
            "pos_y_m": [0., 0., 0.], "pos_z_m": [-1., -1., -1.],
            "speed_xy_mps": [0., 1., 3.], "height_m": [1., 1., 1.],
        })
        gates = pd.DataFrame({"passed": [True, True], "crossing_direction": ["forward", "forward"],
                              "passage_time_s": [0.5, 2.0]})
        segment, metrics = gate_segment(frame, gates)
        self.assertEqual(segment.time_s.tolist(), [0.5, 1., 2.])
        self.assertAlmostEqual(metrics["elapsed_s"], 1.5)
        self.assertAlmostEqual(metrics["speed_xy_time_mean_mps"], 1.25)
        gates.loc[1, "crossing_direction"] = "reverse"
        self.assertEqual(gate_segment(frame, gates), (None, None))
        gates.loc[1, "crossing_direction"] = "forward"
        gates.loc[0, "passed"] = False
        self.assertEqual(gate_segment(frame, gates), (None, None))

    def test_hil_and_sitl_use_distinct_sources_and_reject_duplicate_times(self):
        data = {"timeS": [10., 11.]}
        for prefix, value in (("pos", 1.), ("vel", 2.), ("extPos", 3.), ("extVel", 4.)):
            for i in range(3):
                data[f"{prefix}[{i}]"] = [value, value]
        for i in range(4):
            data[f"quat[{i}]"] = [1. if i == 0 else 0.] * 2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            pd.DataFrame(data).to_csv(path, index=False)
            hil, _ = load_frame(path, "HIL")
            sitl, _ = load_frame(path, "SITL")
            self.assertEqual(hil.pos_x_m.iloc[0], 1.)
            self.assertEqual(sitl.pos_x_m.iloc[0], 3.)
            self.assertEqual(hil.time_s.tolist(), [0., 1.])
            data["timeS"] = [10., 10.]
            pd.DataFrame(data).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "timeS"):
                load_frame(path, "HIL")


if __name__ == "__main__":
    unittest.main()
