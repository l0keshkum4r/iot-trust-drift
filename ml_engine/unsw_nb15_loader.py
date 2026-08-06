"""
UNSW-NB15 Dataset Loader & Preprocessor
========================================
Downloads, caches, and preprocesses UNSW-NB15 training and test datasets.
Encodes categorical features, aligns train/test feature dimensions, and
prepares data for Trust-Drift Fusion (TDF) model training and evaluation.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import OneHotEncoder

TRAIN_URL = (
    "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/"
    "UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/"
    "UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"
)

CATEGORICAL_COLS = ["proto", "service", "state"]
DROP_COLS = ["id", "attack_cat", "label"]


def download_unsw_nb15(output_dir: str) -> Tuple[str, str]:
    """
    Download UNSW-NB15 train and test sets if not already present.
    
    Args:
        output_dir: Local directory path to store CSV files.
        
    Returns:
        Tuple of (train_path, test_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(output_dir, "UNSW_NB15_testing-set.csv")

    if not os.path.exists(train_path):
        print(f"Downloading UNSW-NB15 training set from {TRAIN_URL}...")
        df_train = pd.read_csv(TRAIN_URL)
        df_train.to_csv(train_path, index=False)
        print(f"Saved: {train_path}")

    if not os.path.exists(test_path):
        print(f"Downloading UNSW-NB15 testing set from {TEST_URL}...")
        df_test = pd.read_csv(TEST_URL)
        df_test.to_csv(test_path, index=False)
        print(f"Saved: {test_path}")

    return train_path, test_path


def load_and_preprocess_unsw_nb15(
    data_dir: str = "data",
    subset_ratio: float = 1.0,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Load and preprocess UNSW-NB15 train and test sets.
    
    Args:
        data_dir: Directory to save/load dataset CSV files.
        subset_ratio: Fraction of dataset to use (1.0 for full dataset).
        random_state: Random seed for sampling if subset_ratio < 1.0.
        
    Returns:
        Dictionary containing:
            - X_train: numpy array of preprocessed training features
            - y_train: numpy array of binary training labels (0=normal, 1=attack)
            - X_test: numpy array of preprocessed test features
            - y_test: numpy array of binary test labels
            - feature_cols: list of feature names
            - raw_train: raw DataFrame for training set
            - raw_test: raw DataFrame for test set
            - attack_cat_test: Series of attack category strings for test set
    """
    unsw_dir = os.path.join(data_dir, "unsw_nb15")
    train_path, test_path = download_unsw_nb15(unsw_dir)

    raw_train = pd.read_csv(train_path)
    raw_test = pd.read_csv(test_path)

    if subset_ratio < 1.0:
        raw_train = raw_train.sample(frac=subset_ratio, random_state=random_state).reset_index(drop=True)
        raw_test = raw_test.sample(frac=subset_ratio, random_state=random_state).reset_index(drop=True)

    y_train = raw_train["label"].values.astype(int)
    y_test = raw_test["label"].values.astype(int)
    attack_cat_test = raw_test["attack_cat"].copy()

    # Identify numeric and categorical columns
    feature_candidates = [c for c in raw_train.columns if c not in DROP_COLS]
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_candidates]
    num_cols = [c for c in feature_candidates if c not in cat_cols]

    # Clean numeric columns (fill missing with median)
    for col in num_cols:
        med = raw_train[col].median()
        raw_train[col] = raw_train[col].fillna(med)
        raw_test[col] = raw_test[col].fillna(med)

    # Encode categorical columns with OneHotEncoder
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(raw_train[cat_cols])

    train_cat_encoded = ohe.transform(raw_train[cat_cols])
    test_cat_encoded = ohe.transform(raw_test[cat_cols])

    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()

    # Combine numeric and encoded categorical features
    X_train_num = raw_train[num_cols].values
    X_test_num = raw_test[num_cols].values

    X_train = np.hstack([X_train_num, train_cat_encoded])
    X_test = np.hstack([X_test_num, test_cat_encoded])

    feature_cols = num_cols + cat_feature_names

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "raw_train": raw_train,
        "raw_test": raw_test,
        "attack_cat_test": attack_cat_test,
    }
