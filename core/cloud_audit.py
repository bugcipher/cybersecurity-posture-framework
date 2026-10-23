"""
Cloud Security Posture Auditing Module
Evaluates 7 core cloud security configuration controls:
1. Public bucket exposure (Satisfied)
2. Root-account multi-factor authentication (UNMET)
3. Encryption at rest (Satisfied)
4. Unused IAM role hygiene (UNMET)
5. Overly permissive network ingress rules (Satisfied)
6. Audit logging (Satisfied)
7. Storage versioning (UNMET)
Calculates compliance score (57.14/100) and actionable remediation steps.
"""

from typing import Dict, List, Any, Optional


# 7-item configuration checklist from paper (Section III-F & V-H)
DEFAULT_CLOUD_CONTROLS = [
    {
        "id": "SEC-01",
        "control_name": "Public Bucket Exposure Prevention",
        "description": "Block all public access settings on cloud object storage buckets (S3/GCS/Azure Blob).",
        "status": "PASS",
        "weight": 1.0,
        "mitre_technique": "T1530 - Data from Cloud Storage",
        "remediation": "Maintain automatic public access block and enforce IAM bucket policies."
    },
    {
        "id": "SEC-02",
        "control_name": "Root-Account Multi-Factor Authentication",
        "description": "Enforce hardware or virtual MFA tokens on the tenant root/master administrative account.",
        "status": "FAIL",
        "weight": 1.0,
        "mitre_technique": "T1078.004 - Cloud Accounts",
        "remediation": "URGENT: Enable hardware-based FIDO2/WebAuthn MFA on the root account and lock root API access keys."
    },
    {
        "id": "SEC-03",
        "control_name": "Encryption at Rest",
        "description": "Enforce customer-managed or cloud-provider encryption (AES-256) on all disks and databases.",
        "status": "PASS",
        "weight": 1.0,
        "mitre_technique": "T1486 - Data Encrypted for Impact",
        "remediation": "Ensure KMS key rotation schedules are active every 365 days."
    },
    {
        "id": "SEC-04",
        "control_name": "Unused IAM Role Hygiene",
        "description": "Identify, revoke, and prune IAM roles and service accounts inactive for > 90 days.",
        "status": "FAIL",
        "weight": 1.0,
        "mitre_technique": "T1098 - Account Manipulation",
        "remediation": "Execute automated IAM Access Advisor audit and delete all dormant credentials and shadow admin roles."
    },
    {
        "id": "SEC-05",
        "control_name": "Permissive Network Ingress Rules",
        "description": "Prohibit 0.0.0.0/0 ingress to sensitive ports (e.g., SSH port 22, RDP port 3389, DB port 3306).",
        "status": "PASS",
        "weight": 1.0,
        "mitre_technique": "T1190 - Exploit Public-Facing Application",
        "remediation": "Restrict management ports to corporate VPN CIDRs and bastion jump hosts."
    },
    {
        "id": "SEC-06",
        "control_name": "Comprehensive Audit Logging",
        "description": "Enable immutable control-plane and data-plane logging (CloudTrail, VPC Flow Logs, GCP Audit Logs).",
        "status": "PASS",
        "weight": 1.0,
        "mitre_technique": "T1562.001 - Disable or Modify Tools",
        "remediation": "Export audit trails to an external SIEM with Write-Once-Read-Many (WORM) storage retention."
    },
    {
        "id": "SEC-07",
        "control_name": "Storage Object Versioning",
        "description": "Enable object versioning and MFA-Delete protection on critical storage buckets to prevent ransomware deletion.",
        "status": "FAIL",
        "weight": 1.0,
        "mitre_technique": "T1485 - Data Destruction",
        "remediation": "Enable object versioning across all operational buckets and activate MFA Delete on compliance archives."
    }
]


class CloudSecurityAuditor:
    """Evaluates cloud security controls and calculates compliance posture score."""

    def __init__(self, controls: Optional[List[Dict[str, Any]]] = None):
        self.controls = controls if controls is not None else DEFAULT_CLOUD_CONTROLS

    def audit_controls(self, custom_statuses: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Calculates compliance score as proportion of controls satisfied:
        4 of 7 satisfied = 57.14 / 100.
        """
        controls_evaluated = []
        satisfied_count = 0

        for ctrl in self.controls:
            ctrl_copy = dict(ctrl)
            if custom_statuses and ctrl["id"] in custom_statuses:
                ctrl_copy["status"] = custom_statuses[ctrl["id"]]
            
            if ctrl_copy["status"] == "PASS":
                satisfied_count += 1
            controls_evaluated.append(ctrl_copy)

        total_controls = len(controls_evaluated)
        score = (satisfied_count / total_controls * 100.0) if total_controls > 0 else 0.0
        score = round(score, 2)

        unmet_controls = [c for c in controls_evaluated if c["status"] == "FAIL"]

        return {
            "score": score,
            "total_controls": total_controls,
            "satisfied_count": satisfied_count,
            "unmet_count": total_controls - satisfied_count,
            "controls": controls_evaluated,
            "unmet_controls": unmet_controls,
            "remediation_guidance": [
                f"[{c['id']}] {c['control_name']}: {c['remediation']}" for c in unmet_controls
            ]
        }
