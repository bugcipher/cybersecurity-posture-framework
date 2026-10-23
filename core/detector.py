"""
AI-Based Posture Detection Module (Objective 1)
Implements Random Forest, XGBoost, and custom 4-layer PyTorch Deep Neural Network (DNN).
Evaluates cross-dataset performance on NSL-KDD and UNSW-NB15.
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # type: ignore
import xgboost as xgb  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import TensorDataset, DataLoader  # type: ignore


class DNNClassifier(nn.Module):
    """
    Custom four-layer Deep Neural Network (128-64-32-1) in PyTorch.
    Features: BatchNorm1d, Dropout (p=0.3), ReLU, Sigmoid output.
    """
    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super(DNNClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PostureDetector:
    """Manages training, inference, and evaluation of intrusion detection architectures."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.feature_importances_: Optional[pd.DataFrame] = None

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                            n_estimators: int = 150, n_jobs: int = -1) -> RandomForestClassifier:
        """Trains Random Forest ensemble of 100-200 trees."""
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=n_jobs
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        return rf

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      n_estimators: int = 200, max_depth: int = 6, 
                      learning_rate: float = 0.1) -> xgb.XGBClassifier:
        """Trains XGBoost gradient boosted tree ensemble."""
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        return xgb_model

    def train_dnn(self, X_train: np.ndarray, y_train: np.ndarray,
                  epochs: int = 15, batch_size: int = 256, 
                  lr: float = 0.001) -> DNNClassifier:
        """
        Trains custom 4-layer DNN in PyTorch with Adam optimizer & BCE Loss.
        Mini-batch size 256 for 15 epochs.
        """
        input_dim = X_train.shape[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DNNClassifier(input_dim=input_dim, dropout_rate=0.3).to(device)

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_t, y_t)
        drop_last = len(dataset) > batch_size and (len(dataset) % batch_size == 1)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

        self.models['dnn'] = model
        return model

    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Calculates standard classification metrics: Accuracy, Precision, Recall, F1."""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' has not been trained yet.")

        if len(X_test) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "predictions": np.array([])
            }

        if model_name == 'dnn':
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                probs = model(X_t).cpu().numpy().flatten()
                y_pred = (probs >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100.0
        prec = precision_score(y_test, y_pred, zero_division=0) * 100.0
        rec = recall_score(y_test, y_pred, zero_division=0) * 100.0
        f1 = f1_score(y_test, y_pred, zero_division=0) * 100.0

        return {
            "accuracy": round(acc, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2),
            "predictions": y_pred
        }

    def extract_feature_importances(self, feature_names: List[str], top_n: int = 15) -> pd.DataFrame:
        """Extracts Gini-impurity based feature importances from Random Forest."""
        rf = self.models.get('random_forest')
        if rf is None:
            raise ValueError("Random Forest model is required for Gini feature importance analysis.")

        importances = rf.feature_importances_
        # Ensure length matches
        if len(feature_names) != len(importances):
            if len(feature_names) > len(importances):
                aligned_names = list(feature_names[:len(importances)])
            else:
                aligned_names = list(feature_names) + [f"feature_{i}" for i in range(len(feature_names), len(importances))]
        else:
            aligned_names = list(feature_names)

        df_imp = pd.DataFrame({
            "Feature": aligned_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        
        self.feature_importances_ = df_imp
        return df_imp.head(top_n)

    @staticmethod
    def get_benchmark_results() -> Dict[str, pd.DataFrame]:
        """Returns the exact empirical results from Table I and Table II of the paper."""
        table1_data = {
            "Model": ["Random Forest", "XGBoost"],
            "Acc. (%)": [76.29, 79.65],
            "Prec. (%)": [96.84, 97.03],
            "Recall (%)": [60.32, 66.27],
            "F1 (%)": [74.34, 78.76]
        }
        table2_data = {
            "Model": ["XGBoost", "Deep Neural Network"],
            "Acc. (%)": [90.06, 89.42],
            "Prec. (%)": [98.76, 98.54],
            "Recall (%)": [86.48, 85.72],
            "F1 (%)": [92.22, 91.69]
        }
        return {
            "Table_I_NSL_KDD": pd.DataFrame(table1_data),
            "Table_II_UNSW_NB15": pd.DataFrame(table2_data)
        }
