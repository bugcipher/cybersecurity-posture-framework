"""
Composite Security Scoring Engine (Formula (1) from Research Paper)
Calculates five-component security posture score:
S = w1*D + w2*R + w3*V + w4*I + w5*C
Weights: w1=0.25, w2=0.20, w3=0.20, w4=0.20, w5=0.15 (sum=1.0)
Performs gap analysis (35.7-point disparity) and synthesizes prioritized remediation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd  # type: ignore


class CompositeScorer:
    """Computes five-component organizational security posture score and prioritized mitigations."""

    DEFAULT_WEIGHTS = {
        "D": 0.25,  # Detection quality (F1-based)
        "R": 0.20,  # Forecasted risk-severity
        "V": 0.20,  # Vulnerability exposure (NVD)
        "I": 0.20,  # IAM behavioral risk
        "C": 0.15   # Cloud configuration compliance
    }

    # Case study baseline components matching Table VI
    CASE_STUDY_SCORES = {
        "D": 92.22,
        "R": 66.02,
        "V": 29.25,
        "I": 29.28,
        "C": 57.14
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS
        # Validate weights sum to 1.0
        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 1e-4:
            raise ValueError(f"Weights must sum to 1.0, got {total_w}")

    def calculate_composite_score(self, scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Computes composite security score:
        S = w1*D + w2*R + w3*V + w4*I + w5*C
        """
        s = scores if scores is not None else self.CASE_STUDY_SCORES

        D = float(s.get("D", 92.22))
        R = float(s.get("R", 66.02))
        V = float(s.get("V", 29.25))
        I = float(s.get("I", 29.28))
        C = float(s.get("C", 57.14))

        w1 = self.weights["D"]
        w2 = self.weights["R"]
        w3 = self.weights["V"]
        w4 = self.weights["I"]
        w5 = self.weights["C"]

        contrib_D = round(w1 * D, 3)
        contrib_R = round(w2 * R, 3)
        contrib_V = round(w3 * V, 3)
        contrib_I = round(w4 * I, 3)
        contrib_C = round(w5 * C, 3)

        composite_score = round(contrib_D + contrib_R + contrib_V + contrib_I + contrib_C, 2)

        # Determine Letter Grade & Risk Posture
        if composite_score >= 85.0:
            grade = "A"
            posture_status = "Robust Posture - Low Organizational Risk"
            badge_color = "#10b981"
        elif composite_score >= 70.0:
            grade = "B"
            posture_status = "Adequate Posture - Minor Gaps Identified"
            badge_color = "#3b82f6"
        elif composite_score >= 50.0:
            grade = "C"
            posture_status = "Moderate Risk - Action Required"
            badge_color = "#f59e0b"
        elif composite_score >= 35.0:
            grade = "D"
            posture_status = "High Risk - Immediate Remediations Required"
            badge_color = "#f97316"
        else:
            grade = "F"
            posture_status = "Critical Risk - Severe Exposure Across Controls"
            badge_color = "#ef4444"

        # Gap analysis: detection-only view vs holistic posture
        detection_only_score = D
        gap = round(detection_only_score - composite_score, 2)

        breakdown = [
            {"Component": "Detection (F1-based)", "Symbol": "D", "Score (/100)": D, "Weight": w1, "Weighted Contribution": round(contrib_D, 2)},
            {"Component": "Risk Prediction", "Symbol": "R", "Score (/100)": R, "Weight": w2, "Weighted Contribution": round(contrib_R, 2)},
            {"Component": "Vulnerability (NVD)", "Symbol": "V", "Score (/100)": V, "Weight": w3, "Weighted Contribution": round(contrib_V, 2)},
            {"Component": "IAM Behavioral Risk", "Symbol": "I", "Score (/100)": I, "Weight": w4, "Weighted Contribution": round(contrib_I, 2)},
            {"Component": "Cloud Security Compliance", "Symbol": "C", "Score (/100)": C, "Weight": w5, "Weighted Contribution": round(contrib_C, 2)},
        ]

        return {
            "overall_composite_score": composite_score,
            "grade": grade,
            "posture_status": posture_status,
            "badge_color": badge_color,
            "detection_only_score": detection_only_score,
            "posture_gap": gap,
            "breakdown": breakdown,
            "breakdown_df": pd.DataFrame(breakdown),
            "key_finding": f"A 92.22% detection F1-score masks critical exposures, yielding an overall composite score of {composite_score}/100 (Grade {grade}). The {gap}-point gap demonstrates that detection tools alone systematically overstate organizational security."
        }

    def generate_prioritized_remediation_matrix(self, cve_recommendations: List[Dict[str, Any]],
                                                iam_results: Dict[str, Any],
                                                cloud_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Synthesizes an integrated, severity-ranked remediation plan across all security domains.
        """
        items = []

        # 1. Critical CVEs (CVSS 10.0 and >= 8.0)
        for cve in cve_recommendations:
            items.append({
                "Priority": "P1 - CRITICAL" if cve["CVSS Score"] >= 9.0 else ("P2 - HIGH" if cve["CVSS Score"] >= 7.5 else "P3 - MEDIUM"),
                "Domain": "Vulnerability / CVE",
                "Affected Asset / Control": f"{cve['Asset']} ({cve['CVE ID']})",
                "Deficiency": cve["Summary"],
                "Remediation Action": cve["Actionable Remediation"],
                "Estimated Posture Boost": "+4.5 pts" if cve["CVSS Score"] >= 9.0 else "+2.0 pts"
            })

        # 2. IAM Behavioral Deficiencies
        if iam_results.get("failed_login_ratio_pct", 0) > 15:
            items.append({
                "Priority": "P1 - CRITICAL",
                "Domain": "Identity & Access Management",
                "Affected Asset / Control": "Authentication Service / IdP",
                "Deficiency": f"Abnormal failed login ratio ({iam_results.get('failed_login_ratio_pct')}%) indicates active brute-force or credential stuffing.",
                "Remediation Action": "Implement adaptive rate-limiting, CAPTCHA, and enforce MFA challenge on anomalous IP ranges.",
                "Estimated Posture Boost": "+3.5 pts"
            })
        if iam_results.get("privilege_escalations", 0) > 5:
            items.append({
                "Priority": "P1 - CRITICAL",
                "Domain": "Identity & Access Management",
                "Affected Asset / Control": "Privileged Role Assignments",
                "Deficiency": f"{iam_results.get('privilege_escalations')} privilege escalation attempts detected in access telemetry.",
                "Remediation Action": "Enforce Just-In-Time (JIT) access approval workflows and restrict sudo/admin groups.",
                "Estimated Posture Boost": "+3.0 pts"
            })
        if iam_results.get("atypical_new_location_events", 0) > 5:
            items.append({
                "Priority": "P2 - HIGH",
                "Domain": "Identity & Access Management",
                "Affected Asset / Control": "Geographic Ingress / Session Manager",
                "Deficiency": f"{iam_results.get('atypical_new_location_events')} logins from unverified locations during atypical hours (00:00-05:00).",
                "Remediation Action": "Deploy geographic conditional access policies and alert on impossible travel anomalies.",
                "Estimated Posture Boost": "+2.5 pts"
            })

        # 3. Cloud Configuration Failures
        for ctrl in cloud_results.get("unmet_controls", []):
            priority = "P1 - CRITICAL" if "Root" in ctrl["control_name"] else "P2 - HIGH"
            items.append({
                "Priority": priority,
                "Domain": "Cloud Infrastructure",
                "Affected Asset / Control": f"{ctrl['id']} - {ctrl['control_name']}",
                "Deficiency": ctrl["description"],
                "Remediation Action": ctrl["remediation"],
                "Estimated Posture Boost": "+3.2 pts" if priority == "P1 - CRITICAL" else "+2.1 pts"
            })

        df_remed = pd.DataFrame(items)
        # Sort by Priority P1 > P2 > P3
        priority_order = {"P1 - CRITICAL": 0, "P2 - HIGH": 1, "P3 - MEDIUM": 2}
        df_remed["order"] = df_remed["Priority"].map(priority_order)
        df_remed = df_remed.sort_values(by="order").drop(columns=["order"]).reset_index(drop=True)
        return df_remed
