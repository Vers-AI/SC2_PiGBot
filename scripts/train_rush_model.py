"""
Train logistic regression model for PvZ rush detection.

Purpose: Train ML model from logged game data to classify 12_pool, speedling, none
Key Decisions: Uses -1 for missing values; outputs probability calibrated model
Limitations: Requires ~100+ samples for reliable training

Usage:
    python scripts/train_rush_model.py

Output:
    data/rush_detector_model.pkl
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# Feature columns (must match logging in rush_detection.py)
FEATURE_COLS = [
    "pool_start",
    "nat_start", 
    "gas_time",
    "queen_time",
    "ling_seen",
    "ling_contact",
    "speed_start",
    "ling_has_speed",
    "gas_workers",
    "score_12p",
    "score_speed",
]

# Paths
DATA_DIR = Path("data")
LOG_FILE = DATA_DIR / "rush_detection_log.jsonl"
MODEL_FILE = DATA_DIR / "rush_detector_model.pkl"


def load_data() -> pd.DataFrame:
    """Load and parse JSONL log file."""
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"No log file found at {LOG_FILE}")
    
    records = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {LOG_FILE}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Filter to Zerg games, extract feature matrix X and labels y."""
    df_zerg = df[df['enemy_race'].str.contains('Zerg', case=False, na=False)]
    print(f"Filtered to {len(df_zerg)} Zerg games")
    
    if len(df_zerg) == 0:
        raise ValueError("No Zerg games found in data!")
    
    X = df_zerg[FEATURE_COLS].fillna(-1)
    y = df_zerg['rush_label']
    
    print("\nLabel distribution:")
    print(y.value_counts())
    
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """Train logistic regression, evaluate on held-out test set, print diagnostics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("\n=== Model Evaluation ===")
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)))
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    y_pred = model.predict(X_test)
    print(f"\nTest accuracy: {model.score(X_test, y_test):.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n=== Feature Coefficients (top 5 per class) ===")
    for i, cls in enumerate(model.classes_):
        print(f"\n{cls}:")
        coefs = list(zip(FEATURE_COLS, model.coef_[i]))
        coefs.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, coef in coefs[:5]:
            print(f"  {name}: {coef:+.3f}")
    
    return model


def main():
    print("=== Rush Detection Model Training ===\n")
    
    df = load_data()
    X, y = prepare_features(df)
    
    if len(X) < 20:
        print(f"\nWARNING: Only {len(X)} samples. Model may not be reliable.")
        print("Recommend collecting at least 100 games before training.")
    
    unique_labels = y.unique()
    if len(unique_labels) < 2:
        print(f"\nERROR: Only one class found ({unique_labels[0]}). Need at least 2 classes to train.")
        print("Play more games with different outcomes (rushes and non-rushes).")
        return
    
    model = train_model(X, y)
    
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\n=== Model saved to {MODEL_FILE} ===")
    print(f"Model classes: {list(model.classes_)}")


if __name__ == "__main__":
    main()
