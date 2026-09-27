"""
Preprocessing Pipeline for NSL-KDD Network Intrusion Data
Handles feature taxonomy, categorical one-hot encoding, normalization, and SMOTE balancing.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# Complete 41-feature taxonomy
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
NUMERICAL_FEATURES = [
    c for c in NSL_KDD_COLUMNS if c not in CATEGORICAL_FEATURES and c not in ['label', 'difficulty']
]


def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for one-hot encoding & standard scaling."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )


def balance_with_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) to rebalance minority attack classes.
    """
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X, y)
