

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, classification_report, average_precision_score, roc_auc_score
)
from feature_eng import load_data, prepare_features


def temporal_train_test_split(df, test_months=2):
    df = df.sort_values('timestamp').reset_index(drop=True)
    max_date = df['timestamp'].max()
    cutoff_date = max_date - pd.DateOffset(months=test_months)

    train_df = df[df['timestamp']<= cutoff_date].copy() # seperate dfs
    test_df = df[df['timestamp'] > cutoff_date].copy()

    return train_df, test_df

def train_isolation_forest(X_train, contamination = 0.025):
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples='auto',
        random_state=42,
        n_jobs=1,
        verbose=1
    )

    model.fit(X_train)
    print("Training done!")
    return model

def eval_model(model, X, y, dataset_name="Test"):
    anomaly_scores = model.decision_function(X)
    predictions = model.predict(X)

    y_pred = (predictions == -1).astype(int)
    cm = confusion_matrix(y,y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives (Correct Normal): {cm[0,0]}")
    print(f"False Positives (False Alarms): {cm[0,1]}")
    print(f"False Negatives (Missed Fraud): {cm[1,0]}")
    print(f"True Positives (Caught Fraud): {cm[1,1]}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Normal', 'Fraud']))

    roc_auc = roc_auc_score(y, -anomaly_scores)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    avg_precision = average_precision_score(y, -anomaly_scores)
    print(f"Average Precision Score: {avg_precision:.4f}")

    false_alarm_rate = cm[0,1] / (cm[0,0] + cm[0,1]) * 100
    print(f"False Alarm Rate: {false_alarm_rate:.2f}%")

    fraud_detection_rate = cm[1,1] / (cm[1,0] + cm[1,1]) * 100
    print(f"Fraud Detection Rate: {fraud_detection_rate:.2f}%")

    return {
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'false_alarm_rate': false_alarm_rate,
        'fraud_detection_rate': fraud_detection_rate,
        'anomaly_scores': anomaly_scores,
        'predictions': y_pred
    }

def main():
    df = load_data()

    train_df, test_df = temporal_train_test_split(df, test_months=2)

    X_train, y_train, feature_names, scaler, encoder = prepare_features(
        train_df, 
        fit_scaler=True  # Fit new scaler on training data
    )
    X_test, y_test, _, _, _ = prepare_features(
        test_df, 
        fit_scaler=False,  # Use existing scaler
        scaler=scaler,  # Pass the scaler from training
        fit_encoder=False,  # Use existing encoder
        encoder=encoder  # Pass the encoder from training
    )

    contamination = y_train.mean()
    model = train_isolation_forest(X_train, contamination=contamination)

    train_metrics = eval_model(model, X_train, y_train, "Training")
    test_metrics = eval_model(model, X_test, y_test, "Test")


if __name__ == "__main__": main()