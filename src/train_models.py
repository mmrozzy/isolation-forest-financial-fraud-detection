import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    confusion_matrix, classification_report, average_precision_score, roc_auc_score,
     roc_curve, precision_recall_curve,
)
from feature_eng import load_data, prepare_features
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

def train_one_class_svm(X_train, nu=0.05):
    model = OneClassSVM(
        kernel='rbf',  # Radial Basis Function (Gaussian kernel)
        gamma='auto',  # Kernel coefficient (auto = 1/n_features)
        nu=nu,  # Expected outlier fraction
        cache_size=500,  # Memory cache for kernel (MB)
        verbose=True
    )
    
    print(f"Training on {len(X_train)} samples with nu={nu}...")
    model.fit(X_train)
    
    print("Training complete!")
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

def analyze_feature_importance(model, feature_names, X_train, top_n=15):
    print(f"\nStarting feature importance analysis for {len(feature_names)} features...")
    baseline_scores = model.decision_function(X_train)
    print(f"Calculated baseline scores for {len(X_train)} samples")
    
    importances = []
    for i, feature in enumerate(feature_names):
        X_pert = X_train.copy()
        X_train.iloc[:,i] = np.random.permutation(X_pert.iloc[:,1])
        pert_scores = model.decision_function(X_pert)

        importance = np.abs(pert_scores - baseline_scores).mean()
        importances.append(importance)

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
        
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(top_n)
    
    # Horizontal bar chart
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Most Important Features for Fraud Detection')
    plt.gca().invert_yaxis() 
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved to: models/feature_importance.png")
    
    return feature_importance_df


def main():
    df = load_data()

    train_df, test_df = temporal_train_test_split(df, test_months=2)

    X_train, y_train, feature_names, scaler, encoder = prepare_features(
        train_df, 
        fit_scaler=True
    )
    X_test, y_test, _, _, _ = prepare_features(
        test_df, 
        fit_scaler=False,
        scaler=scaler,
        fit_encoder=False,
        encoder=encoder
    )

    model_if = train_isolation_forest(X_train, contamination=0.05)
    model_svm = train_one_class_svm(X_train, nu=0.05)
    
    train_metrics_if = eval_model(model_if, X_train, y_train, "Training (IF)")
    test_metrics_if = eval_model(model_if, X_test, y_test, "Test (IF)")

    train_metrics_svm = eval_model(model_svm, X_train, y_train, "Training (SVM)")
    test_metrics_svm = eval_model(model_svm, X_test, y_test, "Test (SVM)")
    
    print("\n" + "="*70)
    print(" MODEL COMPARISON")
    print("="*70)
    print("\nIsolation Forest:")
    print(f"  ROC-AUC: {test_metrics_if['roc_auc']:.4f}")
    print(f"  Fraud Detection: {test_metrics_if['fraud_detection_rate']:.2f}%")
    print(f"  False Alarms: {test_metrics_if['false_alarm_rate']:.2f}%")
    
    print("\nOne-Class SVM:")
    print(f"  ROC-AUC: {test_metrics_svm['roc_auc']:.4f}")
    print(f"  Fraud Detection: {test_metrics_svm['fraud_detection_rate']:.2f}%")
    print(f"  False Alarms: {test_metrics_svm['false_alarm_rate']:.2f}%")
    
    print("\n" + "="*70)
    print(" SAVING TRAINED MODEL AND ARTIFACTS")
    print("="*70)
    
    joblib.dump(model_if, 'models/isolation_forest.pkl')
    joblib.dump(model_svm, 'models/one_class_svm.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoder, 'models/encoder.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    import json
    metrics_summary = {
        'isolation_forest': {
            'train': {
                'roc_auc': float(train_metrics_if['roc_auc']),
                'avg_precision': float(train_metrics_if['avg_precision']),
                'false_alarm_rate': float(train_metrics_if['false_alarm_rate']),
                'fraud_detection_rate': float(train_metrics_if['fraud_detection_rate'])
            },
            'test': {
                'roc_auc': float(test_metrics_if['roc_auc']),
                'avg_precision': float(test_metrics_if['avg_precision']),
                'false_alarm_rate': float(test_metrics_if['false_alarm_rate']),
                'fraud_detection_rate': float(test_metrics_if['fraud_detection_rate'])
            }
        },
        'one_class_svm': {
            'train': {
                'roc_auc': float(train_metrics_svm['roc_auc']),
                'avg_precision': float(train_metrics_svm['avg_precision']),
                'false_alarm_rate': float(train_metrics_svm['false_alarm_rate']),
                'fraud_detection_rate': float(train_metrics_svm['fraud_detection_rate'])
            },
            'test': {
                'roc_auc': float(test_metrics_svm['roc_auc']),
                'avg_precision': float(test_metrics_svm['avg_precision']),
                'false_alarm_rate': float(test_metrics_svm['false_alarm_rate']),
                'fraud_detection_rate': float(test_metrics_svm['fraud_detection_rate'])
            }
        }
    }
    
    with open('models/metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)

if __name__ == "__main__": main()