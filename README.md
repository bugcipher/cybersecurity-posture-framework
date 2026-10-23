# AI-Powered Cybersecurity Posture Assessment and Risk Prediction Framework

[![Python](https://img.shields.io/badge/Python-3.14%20%7C%203.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3-00599C.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.60-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Unit Tests](https://img.shields.io/badge/Tests-Passing%20(6%2F6)-10B981.svg)]()
[![Research Status](https://img.shields.io/badge/Research-Replicated%20%26%20Verified-6366F1.svg)]()

> Multi-Source Telemetry Ingestion, Sequence-Aware Risk Forecasting, and Composite Security Scoring.

---

## 💡 The Idea in One Paragraph

Modern enterprises deploy diverse, specialized security controls—firewalls, network intrusion detection systems (IDS/IPS), antivirus, identity and access management (IAM) platforms, and cloud posture tooling—yet operate each in isolation. This siloed visibility leaves leadership with periodic, manual, and reactive audits rather than a continuous, quantitative view of organizational risk. **This framework unifies five distinct security domains into a single composite posture score ($S = 56.54/100$, Grade C):** (1) supervised machine learning and deep learning traffic classifiers, (2) sequence-aware temporal risk forecasting, (3) live NVD vulnerability intelligence, (4) behavioral IAM access auditing, and (5) cloud configuration compliance. Its central empirical finding is that **a strong 92.2% traffic detection F1-score masks critical exposures in unpatched software, access hygiene, and cloud configurations—revealing an operational disparity gap of 35.7 points between single-tool detection and holistic posture.**

---

## 🚦 Project Lifecycle & Phase Status

Every module was engineered and verified sequentially against defined empirical exit criteria:

| Phase | Module Name | Scope & Methodology | State |
|:---:|:---|:---|:---:|
| **0** | **Taxonomy & Formulation** | Feature schema mapping for NSL-KDD (41 features) & UNSW-NB15 (42 features); formulation of 5-component composite posture equation ($S = \sum w_i S_i$). | ✅ Complete |
| **1** | **AI Posture Detection (Obj. 1)** | Binary traffic classification using Random Forest, XGBoost, and a custom PyTorch 4-layer DNN ($128 \to 64 \to 32 \to 1$); cross-dataset evaluation. | ✅ Complete |
| **2** | **Temporal Risk Forecasting (Obj. 2)** | 100-record pseudo-temporal windowing, 4-class risk labeling (Low, Med, High, Crit) via empirical quantiles, 2-layer LSTM and SMOTE-balanced lag-10 Random Forest. | ✅ Complete |
| **3** | **ns-3 Network Simulation** | Controlled discrete-event attacker-victim topology in C++ (10 Mbps, 2 ms delay, DoS burst seconds 8–14); flow feature extraction achieving 100% classification accuracy. | ✅ Complete |
| **4** | **Vulnerability Intelligence** | Live NIST NVD REST API v2.0 integration for enterprise software assets (Apache, MySQL, OpenSSH, Exim) with CVSS base scoring and CVE remediation mapping. | ✅ Complete |
| **5** | **Identity & Cloud Auditing** | Rule-based behavioral analyzer over 500 access logs (20 users) penalizing failed logins, privilege escalations, and atypical hours; 7-point cloud compliance auditor. | ✅ Complete |
| **6** | **Composite Scoring & Gap Analysis** | Weighted five-component scoring engine validating the 35.7-point disparity gap between detection quality ($92.22$) and holistic organizational posture ($56.54$). | ✅ Complete |
| **7** | **Interactive SOC Dashboard** | Strict upload-first Streamlit command center supporting multi-source telemetry ingestion, live telemetry streaming simulation, and prioritized mitigation export. | ✅ Complete |

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Replicate all benchmarks and tables from the research paper (Tables I - VI):
python experiments/run_paper_experiments.py

# 3. Launch the Interactive SOC Command Center:
streamlit run dashboard/app.py
# Open http://localhost:8501 in your browser

# 4. Execute the automated unit test suite:
python -m unittest tests/test_framework.py
```

---

## 📐 Mathematical Formulation & Scoring Model

### 1. Composite Security Posture Formula (Equation 1)
$$\mathbf{S = w_1 D + w_2 R + w_3 V + w_4 I + w_5 C}$$

Where:
- **$D$ (Detection Quality Component):** Defined in terms of F1-score rather than raw detection ratio to prevent evaluation set imbalance from biasing genuine model performance ($w_1 = 0.25$).
- **$R$ (Risk-Severity Component):** Sequence-aware attack intensity forecasting penalized by predicted High- and Critical-risk window frequencies ($w_2 = 0.20$).
- **$V$ (Vulnerability Exposure Component):** Computed as an inverse linear function of the fleet mean CVSS base score retrieved from NVD ($w_3 = 0.20$).
- **$I$ (IAM Behavioral Risk Component):** Rule-based model penalizing elevated failed-login ratios, privilege escalation events, and atypical-hour new-location logins ($w_4 = 0.20$).
- **$C$ (Cloud Configuration Compliance Component):** Proportion of 7 critical cloud infrastructure security controls satisfied ($w_5 = 0.15$).
- **Constraint:** $\sum_{i=1}^5 w_i = 1.00$.

### 2. Case Study Posture Score Breakdown (Table VI)
$$\begin{aligned}
S &= (0.25 \times 92.22) + (0.20 \times 66.02) + (0.20 \times 29.25) + (0.20 \times 29.28) + (0.15 \times 57.14) \\
  &= 23.055 + 13.204 + 5.850 + 5.856 + 8.571 \\
  &= \mathbf{56.54 / 100 \quad (\text{Grade C: Moderate Risk — Action Required})}
\end{aligned}$$

- **Detection-Only View:** $92.22 / 100$
- **True Multi-Dimensional Posture:** $56.54 / 100$
- **Empirical Disparity Gap:** $\mathbf{35.68 \text{ points}}$

### 3. Classification Evaluation Metrics (Equations 2–5)
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \qquad \text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}, \qquad \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 📊 Experimental Results & Benchmark Replication

### Table I: Classifier Performance on NSL-KDD Test Partition (22,544 Records)
| Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Key Observation |
|---|:---:|:---:|:---:|:---:|---|
| **Random Forest** (150 trees) | 76.29 | 96.84 | 60.32 | 74.34 | High precision but lower recall on unseen zero-day test patterns. |
| **XGBoost** (Gradient Boosted Trees) | **79.65** | **97.03** | **66.27** | **78.76** | Improves recall by 6.0% and F1 by 4.4% over Random Forest. |

### Table II: Classifier Performance on UNSW-NB15 Test Partition (175,341 Records)
| Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Key Observation |
|---|:---:|:---:|:---:|:---:|---|
| **XGBoost** | **90.06** | **98.76** | **86.48** | **92.22** | Richer 42-feature telemetry lifts F1 by ~14 points over NSL-KDD. |
| **Deep Neural Network** (PyTorch 128-64-32-1) | 89.42 | 98.54 | 85.72 | 91.69 | Confirms custom deep architecture achieves closely comparable efficacy. |

### Table III: Four-Class Risk-Level Forecasting Performance (100-Record Windows)
| Dataset | Model Architecture | Accuracy (%) | Macro F1 | Critical Class Recall (%) |
|---|---|:---:|:---:|:---:|
| **NSL-KDD** | 2-Layer LSTM (Hidden 32) | 43.2 | 0.15 | 0.0% *(Regression to the mean)* |
| **NSL-KDD** | RF + SMOTE (Lag-10 Features) | 30.8 | 0.22 | 0.0% |
| **UNSW-NB15** | **RF + SMOTE (Lag-10 Features)** | **55.0** | **0.27** | **47.0%** *(High Class Recall: 65%)* |

### Table IV: ns-3 Simulated Traffic Window Statistics (Flow-Level Verification)
| Window Type | $n$ (Windows) | Mean Packets / Window | Mean Packet Size (Bytes) | Classifier Accuracy |
|---|:---:|:---:|:---:|:---:|
| **Normal Traffic** (0–7s, 15–19s) | 10 | 22.3 | 512.0 B | — |
| **Attack Traffic** (8–14s UDP Burst) | 6 | 649.3 | 1005.2 B | — |
| **Held-Out Test Split Validation** | 5 | — | — | **100.0% Accuracy** |

### Table V: NVD Vulnerability Exposure by Representative Enterprise Asset
| Asset Type | Software Component | Mean CVSS Base Score | High-Severity CVEs ($\ge 7.0$) | Maximum Severity Findings |
|---|---|:---:|:---:|:---:|
| **Web Server** | Apache HTTP Server | 7.00 | 3 | 0 |
| **Database** | MySQL Server | 6.50 | 2 | 0 |
| **Remote Access** | OpenSSH Server | **8.00** | **4** | **2 rated at CVSS 10.0** |
| **Mail Server** | Exim SMTP Server | 6.80 | 4 | 1 rated at CVSS 10.0 |
| **Fleet Total** | **All 4 Asset Classes** | **7.08 (Score: 29.25)** | **13** | **3 Critical (CVSS 10.0)** |

### Table VI: Five-Component Composite Security Posture Score Breakdown
| Component | Metric Source | Component Score (/100) | Assigned Weight ($w_i$) | Weighted Contribution |
|---|---|:---:|:---:|:---:|
| **Detection Quality ($D$)** | UNSW-NB15 XGBoost F1 | 92.22 | 0.25 | 23.05 pts |
| **Risk Prediction ($R$)** | Sequence-Aware Penalty Model | 66.02 | 0.20 | 13.20 pts |
| **Vulnerability Exposure ($V$)** | NVD REST API (Mean CVSS 7.08) | 29.25 | 0.20 | 5.85 pts |
| **IAM Behavioral Risk ($I$)** | 500-Record Access Telemetry | 29.28 | 0.20 | 5.86 pts |
| **Cloud Security Compliance ($C$)** | 7-Control Configuration Checklist | 57.14 | 0.15 | 8.57 pts |
| **Overall Composite Score ($S$)** | **Formula (1) Synthesis** | **56.54 / 100** | **$\sum = 1.00$** | **Grade C (Moderate Risk)** |

---

## 🗃️ Repository File Map ("What is Where")

```
core/
  data_loader.py          Robust telemetry ingestor (auto-parses raw headerless NSL-KDD KDDTest+.csv & UNSW-NB15)
  detector.py             Objective 1: Random Forest, XGBoost, PyTorch 4-layer DNN (128-64-32-1), Gini feature ranking
  risk_forecaster.py      Objective 2: 2-layer LSTM, SMOTE-balanced lag-10 Random Forest, quantile risk labeling
  vulnerability_intel.py  Objective 3: Live NIST NVD REST API v2.0 client & calibrated asset intelligence repository
  iam_audit.py            Objective 4: 500-record IAM behavioral model (failed login ratio, priv esc, atypical hours)
  cloud_audit.py          Objective 5: 7-point cloud security misconfiguration auditor (root MFA, versioning, IAM)
  composite_scorer.py     Objective 6: Equation (1) composite scoring engine, disparity gap, prioritized mitigations
simulation/
  attack_simulation.cc    C++ ns-3 discrete-event network simulation topology (3 nodes, 10 Mbps/2 ms, UDP burst)
  ns3_simulator.py        Python flow aggregator (1-second windows) & 100% accuracy Random Forest flow classifier
dashboard/
  app.py                  Interactive Streamlit SOC Command Center with strict data-first upload workflow
experiments/
  run_paper_experiments.pyCLI reproduction driver replicating Tables I through VI in one command
tests/
  test_framework.py       Automated unit test suite verifying mathematical consistency and model behavior (6/6 pass)
data/
  KDDTest+.csv            Official NSL-KDD benchmark test partition (22,544 rows, 43 fields)
  20 Percent Training Set.csv NSL-KDD training split
  Field Names.csv         Standard 41 feature descriptors
  Attack Types.csv        Four primary attack families (DoS, Probe, R2L, U2R)
```

---

## 🌐 Network Simulator 3 (ns-3) Simulation Topology

The discrete-event simulation validates the framework's flow-level feature extraction on independently generated, simulation-based network traffic.

```
+-------------------+                      +-----------------+
|   Attacker Node   |---[ 10 Mbps, 2 ms ]--|                 |
| (5 Mbps UDP burst)|                      |                 |
+-------------------+                      |   Victim Node   |
                                           |  (Packet Sink)  |
+-------------------+                      |                 |
|    Benign Node    |---[ 10 Mbps, 2 ms ]--|                 |
| (100 kbps UDP cst)|                      +-----------------+
+-------------------+
```

- **Benign Node:** Transmits low-rate UDP traffic (100 kbps, 512-byte packets) continuously from 0 to 20 seconds.
- **Attacker Node:** Transmits high-rate UDP traffic (5 Mbps, 1024-byte packets) during simulation seconds 8–14.
- **Source Code:** [`simulation/attack_simulation.cc`](file:///c:/Users/554ra/Downloads/project/simulation/attack_simulation.cc).
- **Compilation (ns-3.35+):**
  ```bash
  cp simulation/attack_simulation.cc /path/to/ns-allinone/ns-3.xx/scratch/
  cd /path/to/ns-allinone/ns-3.xx/
  ./ns3 run "scratch/attack_simulation"
  ```
  Generates `ns3_packet_trace.csv`, parsed by [`simulation/ns3_simulator.py`](file:///c:/Users/554ra/Downloads/project/simulation/ns3_simulator.py) into 1-second flow statistics (packet count, avg packet size, total bytes).

---

## 💻 Programmatic Python API

The core framework is fully modular and can be integrated into custom SOC or SIEM pipelines in a few lines of code:

```python
from core.composite_scorer import CompositeScorer
from core.vulnerability_intel import VulnerabilityIntelligence
from core.iam_audit import IAMAuditor
from core.cloud_audit import CloudSecurityAuditor

# 1. Initialize Composite Scorer
scorer = CompositeScorer(weights={"D": 0.25, "R": 0.20, "V": 0.20, "I": 0.20, "C": 0.15})

# 2. Evaluate Multi-Dimensional Security Components
results = scorer.calculate_composite_score({
    "D": 92.22,  # UNSW-NB15 XGBoost F1 detection score
    "R": 66.02,  # Sequence-aware risk forecasting score
    "V": 29.25,  # NVD vulnerability exposure score (Mean CVSS 7.08)
    "I": 29.28,  # IAM behavioral risk score (26.8% failed logins)
    "C": 57.14   # Cloud compliance score (4/7 controls satisfied)
})

# 3. Output Organizational Posture & Disparity Gap
print(f"Overall Posture Score: {results['overall_composite_score']}/100")  # 56.54
print(f"Assigned Grade:        {results['grade']}")                     # Grade C
print(f"Disparity Gap:         {results['posture_gap']} points")        # 35.68 pts
```

---

## ⚠️ Scientific Limitations (Paper Section VII)

1. **Pseudo-Temporal Windowing Proxy:** Benchmark datasets lack native packet timestamps; 100-record fixed windows serve as a proxy for chronological progression, contributing to weaker Critical-risk recall on NSL-KDD.
2. **Synthetic Audits:** IAM access logs and cloud configuration checklists were modeled from representative threat patterns rather than harvested from production SIEM/CSPM environments.
3. **Simulation Boundary:** The ns-3 topology models a controlled three-node environment and a single bounded DoS burst; live networks experience complex multi-stage APT campaigns and background noise.
4. **NVD Rate Limits:** Unauthenticated NVD API calls constrain rapid inventory scanning; production deployments require authenticated API integration.


