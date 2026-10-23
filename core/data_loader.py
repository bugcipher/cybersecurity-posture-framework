"""
Data Preprocessing, Windowing, and Feature Engineering Module
Supports NSL-KDD and UNSW-NB15 benchmark datasets.
"""

import os
from typing import Tuple, Dict, List, Optional, Any, Union
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
except (ImportError, ModuleNotFoundError):
    class StandardScaler:  # type: ignore
        """Self-contained StandardScaler fallback if scikit-learn is unavailable."""
        def __init__(self):
            self.mean_: Optional[np.ndarray] = None
            self.scale_: Optional[np.ndarray] = None

        def fit(self, X: np.ndarray, y: Optional[Any] = None) -> "StandardScaler":
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0.0] = 1.0
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            if self.mean_ is None or self.scale_ is None:
                raise ValueError("StandardScaler must be fitted before transforming data.")
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
            return self.fit(X, y).transform(X)


NSL_KDD_COLUMNS: List[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

UNSW_NB15_CATEGORICAL: List[str] = ["proto", "service", "state"]


class FallbackLabelEncoder:
    """LabelEncoder that gracefully maps unseen categories at test time to an 'unknown' class."""

    def __init__(self) -> None:
        self.classes_: List[str] = []
        self.mapping_: Dict[str, int] = {}
        self.unknown_index: int = 0

    def fit(self, series: pd.Series) -> "FallbackLabelEncoder":
        unique_vals = list(series.dropna().astype(str).unique())
        if "unknown" not in unique_vals:
            unique_vals.append("unknown")
        self.classes_ = sorted(unique_vals)
        self.mapping_ = {val: idx for idx, val in enumerate(self.classes_)}
        self.unknown_index = self.mapping_["unknown"]
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        return np.array([self.mapping_.get(str(x), self.unknown_index) for x in series], dtype=np.int64)

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        self.fit(series)
        return self.transform(series)


class DataLoader:
    """Loads, preprocesses, and engineers features for intrusion detection & risk forecasting."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, Dict[str, FallbackLabelEncoder]] = {}

    @classmethod
    def load_telemetry_csv(cls, file_or_path: Union[str, bytes, os.PathLike, Any]) -> pd.DataFrame:
        """
        Loads CSV telemetry, automatically identifying whether it is:
        1. Raw NSL-KDD (42 or 43 headerless columns like KDDTest+.csv)
        2. Standard CSV with existing headers
        Ensures binary_label column (0 for normal, 1 for attack).
        """
        if hasattr(file_or_path, 'seek'):
            file_or_path.seek(0)
            
        try:
            df_peek = pd.read_csv(file_or_path, nrows=5)
        except Exception:
            # Fallback if empty or single row
            return pd.DataFrame(columns=NSL_KDD_COLUMNS + ["binary_label"])

        # Check if headerless NSL-KDD
        if len(df_peek.columns) in [42, 43] and (
            isinstance(df_peek.columns[0], int) or 
            df_peek.columns[0] == 0 or 
            str(df_peek.columns[1]).lower() in ['tcp', 'udp', 'icmp']
        ):
            if hasattr(file_or_path, 'seek'):
                file_or_path.seek(0)
            col_names = list(NSL_KDD_COLUMNS)
            if len(df_peek.columns) == 43:
                col_names.append("difficulty_level")
            df = pd.read_csv(file_or_path, header=None, names=col_names)
        else:
            if hasattr(file_or_path, 'seek'):
                file_or_path.seek(0)
            df = pd.read_csv(file_or_path)

        # Ensure binary label
        if 'binary_label' not in df.columns:
            if 'label' in df.columns:
                df['binary_label'] = df['label'].astype(str).str.lower().apply(
                    lambda x: 0 if x in ['normal', '0', 'benign', 'false'] else 1
                )
            elif 'attack_cat' in df.columns:
                df['binary_label'] = df['attack_cat'].astype(str).str.lower().apply(
                    lambda x: 0 if x in ['normal', '0', 'benign', 'false'] else 1
                )
            elif 'class' in df.columns:
                df['binary_label'] = df['class'].astype(str).str.lower().apply(
                    lambda x: 0 if x in ['normal', '0', 'benign', 'false'] else 1
                )
            else:
                df['binary_label'] = 0

        return df

    def generate_synthetic_nsl_kdd(self, n_train: int = 10000, n_test: int = 2500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generates representative NSL-KDD benchmark data reflecting paper characteristics."""
        np.random.seed(self.random_state)

        def create_df(n_samples: int, is_test: bool = False) -> pd.DataFrame:
            protocols = ['tcp', 'udp', 'icmp']
            services = ['http', 'smtp', 'ftp_data', 'private', 'other', 'domain_u', 'eco_i']
            flags = ['SF', 'S0', 'REJ', 'RSTO', 'S1', 'SH']

            # Label distribution (attack vs normal)
            attack_prob = 0.53 if is_test else 0.46
            is_attack = np.random.binomial(1, attack_prob, n_samples)

            data = {
                "duration": np.where(is_attack, np.random.exponential(1.5, n_samples), np.random.exponential(0.2, n_samples)),
                "protocol_type": np.random.choice(protocols, n_samples, p=[0.75, 0.18, 0.07]),
                "service": np.random.choice(services, n_samples),
                "flag": np.random.choice(flags, n_samples),
                "src_bytes": np.where(is_attack, np.random.lognormal(7.5, 2.0, n_samples), np.random.lognormal(5.0, 1.2, n_samples)),
                "dst_bytes": np.where(is_attack, np.random.lognormal(6.5, 2.2, n_samples), np.random.lognormal(6.0, 1.5, n_samples)),
                "land": np.zeros(n_samples, dtype=float),
                "wrong_fragment": np.where(is_attack, np.random.choice([0, 1, 3], n_samples, p=[0.9, 0.08, 0.02]), 0),
                "urgent": np.zeros(n_samples, dtype=float),
                "hot": np.where(is_attack, np.random.poisson(0.5, n_samples), 0),
                "num_failed_logins": np.where(is_attack, np.random.choice([0, 1, 2], n_samples, p=[0.92, 0.06, 0.02]), 0),
                "logged_in": np.where(is_attack, np.random.binomial(1, 0.15, n_samples), np.random.binomial(1, 0.70, n_samples)),
                "num_compromised": np.where(is_attack, np.random.poisson(0.2, n_samples), 0),
                "root_shell": np.zeros(n_samples, dtype=float),
                "su_attempted": np.zeros(n_samples, dtype=float),
                "num_root": np.zeros(n_samples, dtype=float),
                "num_file_creations": np.zeros(n_samples, dtype=float),
                "num_shells": np.zeros(n_samples, dtype=float),
                "num_access_files": np.zeros(n_samples, dtype=float),
                "num_outbound_cmds": np.zeros(n_samples, dtype=float),
                "is_host_login": np.zeros(n_samples, dtype=float),
                "is_guest_login": np.zeros(n_samples, dtype=float),
                "count": np.where(is_attack, np.random.randint(10, 300, n_samples), np.random.randint(1, 40, n_samples)),
                "srv_count": np.where(is_attack, np.random.randint(5, 250, n_samples), np.random.randint(1, 35, n_samples)),
                "serror_rate": np.where(is_attack, np.random.uniform(0.3, 1.0, n_samples), np.random.uniform(0.0, 0.05, n_samples)),
                "srv_serror_rate": np.where(is_attack, np.random.uniform(0.3, 1.0, n_samples), np.random.uniform(0.0, 0.05, n_samples)),
                "rerror_rate": np.random.uniform(0.0, 0.1, n_samples),
                "srv_rerror_rate": np.random.uniform(0.0, 0.1, n_samples),
                "same_srv_rate": np.where(is_attack, np.random.uniform(0.05, 0.8, n_samples), np.random.uniform(0.85, 1.0, n_samples)),
                "diff_srv_rate": np.where(is_attack, np.random.uniform(0.2, 0.9, n_samples), np.random.uniform(0.0, 0.15, n_samples)),
                "srv_diff_host_rate": np.random.uniform(0.0, 0.2, n_samples),
                "dst_host_count": np.random.randint(1, 255, n_samples),
                "dst_host_srv_count": np.where(is_attack, np.random.randint(1, 100, n_samples), np.random.randint(100, 255, n_samples)),
                "dst_host_same_srv_rate": np.where(is_attack, np.random.uniform(0.05, 0.6, n_samples), np.random.uniform(0.8, 1.0, n_samples)),
                "dst_host_diff_srv_rate": np.where(is_attack, np.random.uniform(0.1, 0.8, n_samples), np.random.uniform(0.0, 0.1, n_samples)),
                "dst_host_same_src_port_rate": np.random.uniform(0.0, 0.3, n_samples),
                "dst_host_srv_diff_host_rate": np.random.uniform(0.0, 0.1, n_samples),
                "dst_host_serror_rate": np.where(is_attack, np.random.uniform(0.2, 1.0, n_samples), 0.0),
                "dst_host_srv_serror_rate": np.where(is_attack, np.random.uniform(0.2, 1.0, n_samples), 0.0),
                "dst_host_rerror_rate": np.random.uniform(0.0, 0.1, n_samples),
                "dst_host_srv_rerror_rate": np.random.uniform(0.0, 0.1, n_samples),
                "label": np.where(is_attack == 1, 'neptune', 'normal')
            }
            if is_test:
                unseen_idx = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
                for idx in unseen_idx:
                    if data["label"][idx] != 'normal':
                        data["label"][idx] = 'mscan'
                        data["service"][idx] = 'sunrpc'

            df = pd.DataFrame(data)
            df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
            return df

        train_df = create_df(n_train, is_test=False)
        test_df = create_df(n_test, is_test=True)
        return train_df, test_df

    def generate_synthetic_unsw_nb15(self, n_train: int = 10000, n_test: int = 2500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generates representative UNSW-NB15 benchmark data with 42 features."""
        np.random.seed(self.random_state + 10)

        feature_names = [
            "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
            "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
            "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
            "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
            "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
            "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
            "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"
        ]

        def create_df(n_samples: int) -> pd.DataFrame:
            is_attack = np.random.binomial(1, 0.45, n_samples)
            protos = ['tcp', 'udp', 'unas', 'arp', 'ospf']
            services = ['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh']
            states = ['FIN', 'CON', 'INT', 'REQ', 'RST']

            data: Dict[str, Any] = {
                'proto': np.random.choice(protos, n_samples),
                'service': np.random.choice(services, n_samples),
                'state': np.random.choice(states, n_samples)
            }

            for col in feature_names:
                if col in ['proto', 'service', 'state']:
                    continue
                if 'bytes' in col or 'spkts' in col or 'dpkts' in col:
                    data[col] = np.where(is_attack, np.random.lognormal(6.0, 1.8, n_samples), np.random.lognormal(4.5, 1.2, n_samples))
                elif 'rate' in col or 'load' in col:
                    data[col] = np.where(is_attack, np.random.exponential(5000, n_samples), np.random.exponential(1000, n_samples))
                elif 'ttl' in col:
                    data[col] = np.where(is_attack, np.random.choice([64, 254], n_samples), np.random.choice([64, 32], n_samples))
                else:
                    data[col] = np.random.uniform(0, 10, n_samples)

            data['label'] = is_attack
            data['attack_cat'] = np.where(is_attack == 0, 'Normal', np.random.choice(
                ['Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms'],
                n_samples
            ))
            df = pd.DataFrame(data)
            df['binary_label'] = df['label']
            return df

        train_df = create_df(n_train)
        test_df = create_df(n_test)
        return train_df, test_df

    def preprocess_dataset(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           categorical_cols: List[str], target_col: str = 'binary_label',
                           dataset_name: str = 'dataset') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Preprocesses training and testing splits: label encoding + standardization."""
        train_df = train_df.copy()
        test_df = test_df.copy()

        encoders = {}
        for col in categorical_cols:
            if col in train_df.columns:
                le = FallbackLabelEncoder()
                train_df[col] = le.fit_transform(train_df[col])
                test_df[col] = le.transform(test_df[col])
                encoders[col] = le
        self.encoders[dataset_name] = encoders

        # Drop raw non-feature columns
        cols_to_drop = [target_col, 'label', 'attack_cat', 'binary_label']
        feature_cols = [c for c in train_df.columns if c not in cols_to_drop]

        X_train = train_df[feature_cols].fillna(0).values.astype(np.float32)
        y_train = train_df[target_col].fillna(0).values.astype(np.int64)
        X_test = test_df[feature_cols].fillna(0).values.astype(np.float32)
        y_test = test_df[target_col].fillna(0).values.astype(np.int64)

        # Standardize numeric features (zero mean, unit variance)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[dataset_name] = scaler

        return X_train_scaled, y_train, X_test_scaled, y_test, feature_cols

    @staticmethod
    def create_pseudo_temporal_windows(predictions_or_labels: np.ndarray, 
                                       window_size: int = 100) -> pd.DataFrame:
        """
        Partitions sequential records into fixed-size pseudo-temporal windows (100 records/window).
        Counts predicted/actual attack connections per window.
        """
        n_windows = len(predictions_or_labels) // window_size
        records = []
        for w in range(n_windows):
            window_slice = predictions_or_labels[w * window_size : (w + 1) * window_size]
            attack_count = int(np.sum(window_slice == 1))
            records.append({
                "window_idx": w,
                "window_size": window_size,
                "attack_count": attack_count,
                "attack_ratio": attack_count / float(window_size)
            })
        if not records:
            return pd.DataFrame(columns=["window_idx", "window_size", "attack_count", "attack_ratio"])
        return pd.DataFrame(records)

    @staticmethod
    def assign_risk_quantiles(window_df: pd.DataFrame, 
                              q_thresholds: Optional[Tuple[float, float, float]] = None) -> Tuple[pd.DataFrame, Tuple[float, float, float]]:
        """
        Assigns empirical quantile thresholds (33rd, 66th, 90th percentiles) to attack counts
        categorizing each window into 4 risk levels:
        0: Low, 1: Medium, 2: High, 3: Critical.
        """
        df = window_df.copy()
        if df.empty or 'attack_count' not in df.columns:
            df['risk_level'] = pd.Series(dtype=int)
            df['risk_category'] = pd.Series(dtype=str)
            return df, (0.0, 0.0, 0.0) if q_thresholds is None else q_thresholds

        if q_thresholds is None:
            q33 = float(df['attack_count'].quantile(0.33))
            q66 = float(df['attack_count'].quantile(0.66))
            q90 = float(df['attack_count'].quantile(0.90))
            q_thresholds = (q33, q66, q90)
        else:
            q33, q66, q90 = q_thresholds

        def get_risk(count: float) -> int:
            if count <= q33:
                return 0  # Low
            elif count <= q66:
                return 1  # Medium
            elif count <= q90:
                return 2  # High
            else:
                return 3  # Critical

        df['risk_level'] = df['attack_count'].apply(get_risk).astype(int)
        risk_map = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
        df['risk_category'] = df['risk_level'].map(risk_map).fillna('Low')
        return df, q_thresholds

    @staticmethod
    def build_lag_features(window_df: pd.DataFrame, lookback: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Constructs lag-1 through lag-10 features for predicting next window risk category.
        """
        df = window_df.copy()
        lag_cols: List[str] = []
        for i in range(1, lookback + 1):
            col_name = f"lag_{i}"
            df[col_name] = df['attack_count'].shift(i)
            lag_cols.append(col_name)

        df = df.dropna().reset_index(drop=True)
        if df.empty:
            return np.empty((0, lookback), dtype=np.float32), np.empty((0,), dtype=np.int64), lag_cols

        X_lag = df[lag_cols].values.astype(np.float32)
        y_target = df['risk_level'].values.astype(np.int64)
        return X_lag, y_target, lag_cols
