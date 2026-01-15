# Fraud Detection Exercise - Work in Progress

Project to explore machine learning-based fraud detection with synthetic transaction data. Work in progress.

## What's This?

This is a personal learning exercise that goes through the complete workflow of building a fraud detection system:
- Generating synthetic transaction data
- Training an ML anomaly detection model
- Building a real-time monitoring dashboard

## Project Structure

```
fraud-detection-exercise/
├── data/
│   └── transactions.csv          # Synthetic transaction data
├── models/
│   ├── isolation_forest.pkl      # Trained model
│   ├── scaler.pkl                # Feature scaler
│   ├── encoder.pkl               # Category encoder
│   └── feature_names.pkl         # Feature list
├── src/
│   ├── generate_data.py          # Create synthetic transactions
│   ├── feature_eng.py            # Feature engineering utilities
│   ├── train_models.py           # Train the detection model
│   └── dashboard.py              # Streamlit dashboard
```

## How It Works

**Data Generation**: Uses Faker to create fraudulent and non-fruadulent realistic transaction data including amounts, merchant categories, locations, and timestamps.

**Model**: Isolation Forest algorithm detects anomalies by identifying transactions that don't fit normal patterns. One-class svm also coded, but yields worse performance.

**Dashboard**: Streamlit app simulates real-time transaction processing, showing predictions, statistics, and fraud alerts (to be added).

## Skills Learned

- Working with synthetic data for ML prototyping
- Implementing anomaly detection with scikit-learn
- Feature engineering for transaction data
- Building interactive dashboards with Streamlit
- Managing ML model artifacts (pickling/loading)

## Tech Stack

- **Python**
- **pandas** - Data manipulation
- **scikit-learn** - ML models
- **Streamlit** - Dashboard interface
- **Faker** - Synthetic data generation

## Notes

This is an educational project using synthetic data. Real-world fraud detection requires:
- Much larger datasets
- More sophisticated features
- Handling imbalanced data
- Regular model retraining
- Integration with existing systems

## Model Results

Two anomaly detection models were trained and evaluated on the synthetic transaction data:

### Isolation Forest (Primary Model)
- **Test ROC-AUC**: 93.59%
- **Fraud Detection Rate**: 80.21%
- **False Alarm Rate**: 6.93%
- **Average Precision**: **58.86%**

### One-Class SVM (Baseline)
- **Test ROC-AUC**: 85.16%
- **Fraud Detection Rate**: 61.46%
- **False Alarm Rate**: 7.31%
- **Average Precision**: 22.14%

The Isolation Forest model significantly outperforms One-Class SVM across all metrics, making it the chosen approach for the dashboard. The model achieves a good balance between detecting fraud (80% detection rate) and minimizing false alarms (~7%).

**Note**: Currently working on improving model performance through feature engineering and hyperparameter tuning.

## Next Steps

To be added:
- Fraud Alert Panel
- Visualizations/Charts on dash
- CSS styling
- transaction search, filter, export data

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data

Create synthetic transaction data with realistic patterns:

```bash
python src/generate_data.py
```

This generates ~10,000 transactions with fraud cases mixed in.

### 3. Train the Model

Train an Isolation Forest model to detect anomalies:

```bash
python src/train_models.py
```

### 4. Launch the Dashboard

See the model in action with a real-time streaming dashboard:

```bash
streamlit run src/dashboard.py
```
