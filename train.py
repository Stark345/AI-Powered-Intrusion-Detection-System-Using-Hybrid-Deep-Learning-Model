"""
Standalone Training Pipeline for Hybrid CNN-LSTM Network Intrusion Detection
Downloads NSL-KDD benchmark data, applies SMOTE, trains model, and outputs metrics.
"""
import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.model import build_hybrid_model
from src.preprocess import NSL_KDD_COLUMNS, build_preprocessor, balance_with_smote

DATA_DIR = "./data"
TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

    if not os.path.exists(train_path):
        print("[INFO] Downloading KDDTrain+.txt...")
        urllib.request.urlretrieve(TRAIN_URL, train_path)
    if not os.path.exists(test_path):
        print("[INFO] Downloading KDDTest+.txt...")
        urllib.request.urlretrieve(TEST_URL, test_path)

    return train_path, test_path


def main():
    print("🚀 Initializing Hybrid CNN-LSTM Intrusion Detection Training Pipeline...")
    train_path, test_path = download_data()

    print("[INFO] Loading datasets...")
    train_df = pd.read_csv(train_path, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, names=NSL_KDD_COLUMNS)

    # Binary label mapping (normal vs attack)
    y_train = (train_df['label'] != 'normal').astype(int)
    y_test = (test_df['label'] != 'normal').astype(int)

    X_train = train_df.drop(columns=['label', 'difficulty'])
    X_test = test_df.drop(columns=['label', 'difficulty'])

    print("[INFO] Fitting preprocessor and transforming features...")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print("[INFO] Applying SMOTE class balancing...")
    X_train_bal, y_train_bal = balance_with_smote(X_train_proc, y_train.values)

    # Reshape for 1D CNN: (samples, features, 1)
    X_train_dl = X_train_bal.reshape(X_train_bal.shape[0], X_train_bal.shape[1], 1)
    X_test_dl = X_test_proc.reshape(X_test_proc.shape[0], X_test_proc.shape[1], 1)

    print(f"[INFO] Building Hybrid Model (Input features: {X_train_bal.shape[1]})...")
    model = build_hybrid_model(input_features=X_train_bal.shape[1])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    print("[INFO] Training model...")
    model.fit(
        X_train_dl, y_train_bal,
        epochs=15,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    print("\n📊 Evaluating on Test Set (KDDTest+)...")
    y_pred_prob = model.predict(X_test_dl)
    y_pred = (y_pred_prob > 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    model.save("ids_hybrid_model.h5")
    print("✅ Model weights saved to ids_hybrid_model.h5")


if __name__ == "__main__":
    main()
