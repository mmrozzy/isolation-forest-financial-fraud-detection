import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, classification_report, average_precision_score, roc_auc_score,
     roc_curve, precision_recall_curve,
)
from feature_eng import load_data, prepare_features
import matplotlib.pyplot as plt
import seaborn as sns


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

# CLAUDE GENERATED - TO BE SEEN IF USEFUL
def plot_evaluation_metrics(train_metrics, test_metrics, y_train, y_test):
    
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ==================== PLOT 1: Training Confusion Matrix ====================
    sns.heatmap(
        train_metrics['confusion_matrix'],  # The 2x2 matrix
        annot=True,  # Show numbers inside cells
        fmt='d',  # Format as integers (not decimals)
        cmap='Blues',  # Color scheme
        ax=axes[0,0]  # Top-left subplot
    )
    axes[0,0].set_title('Confusion Matrix - Training Set')
    axes[0,0].set_ylabel('True Label')
    axes[0,0].set_xlabel('Predicted Label')
    axes[0,0].set_xticklabels(['Normal', 'Fraud'])
    axes[0,0].set_yticklabels(['Normal', 'Fraud'])
    
    # ==================== PLOT 2: Test Confusion Matrix ====================
    sns.heatmap(
        test_metrics['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        ax=axes[0,1]  # Top-right subplot
    )
    axes[0,1].set_title('Confusion Matrix - Test Set')
    axes[0,1].set_ylabel('True Label')
    axes[0,1].set_xlabel('Predicted Label')
    axes[0,1].set_xticklabels(['Normal', 'Fraud'])
    axes[0,1].set_yticklabels(['Normal', 'Fraud'])
    
    # ==================== PLOT 3: ROC Curves ====================
    fpr_train, tpr_train, _ = roc_curve(y_train, -train_metrics['anomaly_scores'])

    
    # Calculate ROC curve points for test data
    fpr_test, tpr_test, _ = roc_curve(y_test, -test_metrics['anomaly_scores'])
    
    # Plot training ROC curve
    axes[1,0].plot(
        fpr_train, tpr_train, 
        label=f'Train (AUC = {train_metrics["roc_auc"]:.3f})', 
        linewidth=2
    )
    
    # Plot test ROC curve
    axes[1,0].plot(
        fpr_test, tpr_test, 
        label=f'Test (AUC = {test_metrics["roc_auc"]:.3f})', 
        linewidth=2
    )
    
    # Plot diagonal line (random classifier baseline)
    axes[1,0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    axes[1,0].set_xlabel('False Positive Rate')
    axes[1,0].set_ylabel('True Positive Rate')
    axes[1,0].set_title('ROC Curve')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Calculate Precision-Recall curve points
    precision_train, recall_train, _ = precision_recall_curve(
        y_train, -train_metrics['anomaly_scores']
    )
    precision_test, recall_test, _ = precision_recall_curve(
        y_test, -test_metrics['anomaly_scores']
    )
    
    # Plot training PR curve
    axes[1,1].plot(
        recall_train, precision_train, 
        label=f'Train (AP = {train_metrics["avg_precision"]:.3f})', 
        linewidth=2
    )
    
    # Plot test PR curve
    axes[1,1].plot(
        recall_test, precision_test, 
        label=f'Test (AP = {test_metrics["avg_precision"]:.3f})', 
        linewidth=2
    )
    
    axes[1,1].set_xlabel('Recall')
    axes[1,1].set_ylabel('Precision')
    axes[1,1].set_title('Precision-Recall Curve')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout() 
    plt.savefig('models/model_evaluation.png', dpi=300, bbox_inches='tight')
    print("\nEvaluation plots saved to: models/model_evaluation.png")
    plt.show()

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

    plot_evaluation_metrics(train_metrics, test_metrics, y_train, y_test)


if __name__ == "__main__": main()