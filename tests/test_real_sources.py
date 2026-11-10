import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_real_sitl_logs import normalize_log_frame


class RealSourcesTests(unittest.TestCase):
    def test_external_sources_required_for_real_and_sitl_fallback_preserved(self):
        data = {"timeS": [0., 1., 2.]}
        for prefix, value in (("pos", 1.), ("vel", 2.), ("extPos", 3.), ("extVel", 4.)):
            for i in range(3):
                data[f"{prefix}[{i}]"] = [value] * 3
        for i in range(4):
            data[f"quat[{i}]"] = [1. if i == 0 else 0.] * 3
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            pd.DataFrame(data).to_csv(path, index=False)
            frame, profile = normalize_log_frame(path, domain="real")
            self.assertEqual((profile["position_prefix"], profile["velocity_prefix"]), ("extPos", "extVel"))
            self.assertEqual((frame.pos_x_m.iloc[0], frame.vel_x_mps.iloc[0]), (3., 4.))
            pd.DataFrame(data).drop(columns="extVel[2]").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "extVel"):
                normalize_log_frame(path, domain="real")
            _, sitl_profile = normalize_log_frame(path, domain="sitl")
            self.assertEqual(sitl_profile["velocity_prefix"], "vel")


if __name__ == "__main__":
    unittest.main()
