"""
Generates clean sample data files for network traffic and IAM access logs
so users can easily upload and test the dynamic posture assessment.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

os.makedirs("sample_data", exist_ok=True)

# 1. Sample Network Traffic (NSL-KDD style features)
np.random.seed(42)
n_samples = 500
protocols = ['tcp', 'udp', 'icmp']
services = ['http', 'smtp', 'ftp_data', 'private', 'other']
flags = ['SF', 'S0', 'REJ', 'RSTO']
is_attack = np.random.binomial(1, 0.45, n_samples)

net_data = {
    "duration": np.where(is_attack, np.random.exponential(1.2, n_samples), np.random.exponential(0.1, n_samples)),
    "protocol_type": np.random.choice(protocols, n_samples),
    "service": np.random.choice(services, n_samples),
    "flag": np.random.choice(flags, n_samples),
    "src_bytes": np.where(is_attack, np.random.randint(500, 50000, n_samples), np.random.randint(10, 1500, n_samples)),
    "dst_bytes": np.where(is_attack, np.random.randint(500, 40000, n_samples), np.random.randint(10, 2000, n_samples)),
    "logged_in": np.where(is_attack, np.random.binomial(1, 0.2, n_samples), np.random.binomial(1, 0.8, n_samples)),
    "count": np.where(is_attack, np.random.randint(20, 250, n_samples), np.random.randint(1, 30, n_samples)),
    "srv_count": np.where(is_attack, np.random.randint(10, 200, n_samples), np.random.randint(1, 25, n_samples)),
    "same_srv_rate": np.where(is_attack, np.random.uniform(0.1, 0.7, n_samples), np.random.uniform(0.8, 1.0, n_samples)),
    "dst_host_srv_count": np.where(is_attack, np.random.randint(1, 80, n_samples), np.random.randint(100, 255, n_samples)),
    "dst_host_same_srv_rate": np.where(is_attack, np.random.uniform(0.05, 0.6, n_samples), np.random.uniform(0.8, 1.0, n_samples)),
    "label": np.where(is_attack == 1, 'attack', 'normal')
}
df_net = pd.DataFrame(net_data)
df_net.to_csv("sample_data/sample_network_traffic.csv", index=False)
print("Saved sample_data/sample_network_traffic.csv")

# 2. Sample IAM Access Logs
users = [f"user_{i:02d}" for i in range(1, 21)]
records = []
base_time = datetime(2026, 10, 1, 8, 0, 0)
for i in range(200):
    user = np.random.choice(users)
    hour = np.random.randint(0, 24)
    is_atypical = 1 if 0 <= hour < 5 else 0
    is_failed = np.random.binomial(1, 0.25)
    is_priv = np.random.binomial(1, 0.08)
    records.append({
        "timestamp": (base_time + timedelta(hours=i*2, minutes=np.random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user,
        "action": "privilege_escalation" if is_priv else ("login_attempt"),
        "status": "FAILURE" if is_failed else "SUCCESS",
        "location": np.random.choice(["US-East", "US-West", "EU-Central", "CN-Beijing"]),
        "is_new_location": np.random.binomial(1, 0.15),
        "hour": hour
    })
df_iam = pd.DataFrame(records)
df_iam.to_csv("sample_data/sample_iam_access_logs.csv", index=False)
print("Saved sample_data/sample_iam_access_logs.csv")
