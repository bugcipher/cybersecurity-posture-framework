"""
Unit Test Suite for Cybersecurity Posture Assessment Framework
Validates detection, forecasting, vulnerability scoring, IAM auditing, cloud compliance,
ns-3 simulation, and composite scoring formulas.
"""

import unittest
import numpy as np  # type: ignore

from core.data_loader import DataLoader
from core.detector import PostureDetector
from core.risk_forecaster import RiskForecaster
from core.vulnerability_intel import VulnerabilityIntelligence
from core.iam_audit import IAMAuditor
from core.cloud_audit import CloudSecurityAuditor
from core.composite_scorer import CompositeScorer
from simulation.ns3_simulator import NS3Simulator


class TestCybersecurityFramework(unittest.TestCase):

    def setUp(self):
        self.data_loader = DataLoader(random_state=42)
        self.detector = PostureDetector(random_state=42)
        self.forecaster = RiskForecaster(random_state=42)
        self.v_intel = VulnerabilityIntelligence()
        self.iam_auditor = IAMAuditor(random_state=42)
        self.cloud_auditor = CloudSecurityAuditor()
        self.scorer = CompositeScorer()
        self.ns3_sim = NS3Simulator(random_state=42)

    def test_composite_scoring_formula(self):
        """Verifies Equation (1) and case study values matching Table VI."""
        res = self.scorer.calculate_composite_score({
            "D": 92.22,
            "R": 66.02,
            "V": 29.25,
            "I": 29.28,
            "C": 57.14
        })
        # 0.25*92.22 + 0.20*66.02 + 0.20*29.25 + 0.20*29.28 + 0.15*57.14 = 56.536
        self.assertAlmostEqual(res["overall_composite_score"], 56.53, delta=0.05)
        self.assertEqual(res["grade"], "C")
        self.assertAlmostEqual(res["posture_gap"], 35.7, delta=0.2)

    def test_vulnerability_score(self):
        """Verifies NVD aggregate vulnerability score matches Table V and VI."""
        v_score = self.v_intel.calculate_vulnerability_score()
        self.assertEqual(v_score, 29.25)
        table5 = self.v_intel.get_asset_exposure_table()
        self.assertEqual(len(table5), 4)
        remeds = self.v_intel.get_remediation_guidance(min_cvss=7.0)
        self.assertGreaterEqual(len(remeds), 10)

    def test_iam_auditor(self):
        """Verifies IAM rule-based model on simulated access telemetry."""
        logs = self.iam_auditor.generate_simulated_access_logs(500, 20)
        self.assertEqual(len(logs), 500)
        iam_res = self.iam_auditor.evaluate_iam_risk(logs)
        self.assertEqual(iam_res["failed_login_ratio_pct"], 26.8)
        self.assertEqual(iam_res["privilege_escalations"], 15)
        self.assertEqual(iam_res["atypical_new_location_events"], 16)
        self.assertEqual(iam_res["score"], 29.28)

    def test_cloud_auditor(self):
        """Verifies 7 cloud security controls checklist and score (4/7 = 57.14%)."""
        res = self.cloud_auditor.audit_controls()
        self.assertEqual(res["total_controls"], 7)
        self.assertEqual(res["satisfied_count"], 4)
        self.assertEqual(res["score"], 57.14)
        self.assertEqual(len(res["unmet_controls"]), 3)

    def test_ns3_simulation_flow_stats(self):
        """Verifies ns-3 1-second flow aggregation and 100% classifier accuracy."""
        self.ns3_sim.run_simulation()
        df_win = self.ns3_sim.aggregate_one_second_windows()
        self.assertEqual(len(df_win), 20)
        eval_res = self.ns3_sim.train_flow_classifier()
        self.assertEqual(eval_res["accuracy"], 100.0)

    def test_pseudo_temporal_windowing(self):
        """Verifies windowing and quantile threshold calculation for risk forecasting."""
        fake_labels = np.array([0, 1, 0, 1, 1, 0] * 50)  # 300 records
        df_win = self.data_loader.create_pseudo_temporal_windows(fake_labels, window_size=50)
        self.assertEqual(len(df_win), 6)
        df_risk, q_thresh = self.data_loader.assign_risk_quantiles(df_win)
        self.assertIn("risk_category", df_risk.columns)
        self.assertEqual(len(q_thresh), 3)


if __name__ == "__main__":
    unittest.main()
