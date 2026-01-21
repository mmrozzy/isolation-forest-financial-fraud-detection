import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


FILEPATH = os.path.join('data', 'transactions.csv')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

LARGE_TIME_DELTA_MINUTES = 10000
RAPID_TRANSACTION_THRESHOLD_MINUTES = 5

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _sort_by_user_and_time(df):
    df = df.copy()
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    return df


def load_data(filepath=FILEPATH):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def create_time_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek 
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = (
        (df['hour'] >= 23) | (df['hour'] <= 6)
    ).astype(int)  # can be customized later
    df['is_business_hours'] = (
        (df['hour'] >= 9) & (df['hour'] <= 17)
    ).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df


def create_user_features(df):
    df = _sort_by_user_and_time(df)
    df['user_transaction_count'] = df.groupby('user_id').cumcount()
    # Shift to exclude current entry (use previous transactions for comparison)
    df['user_avg_amount'] = df.groupby('user_id')['amount'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df['user_std_amount'] = df.groupby('user_id')['amount'].transform(
        lambda x: x.expanding().std().shift(1)
    )
    df['user_avg_amount'] = df['user_avg_amount'].fillna(df['amount'].mean())
    df['user_std_amount'] = df['user_std_amount'].fillna(df['amount'].std())
    df['amount_deviation'] = (
        (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1)
    )
    df['amount_zscore'] = np.abs(
        (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1)
    )
    df['user_median_amount'] = df.groupby('user_id')['amount'].transform(
        lambda x: x.expanding().median().shift(1)
    )
    df['user_median_amount'] = df['user_median_amount'].fillna(
        df['amount'].median()
    )
    df['amount_vs_median'] = df['amount'] / (df['user_median_amount'] + 1)
    df['user_max_amount'] = df.groupby('user_id')['amount'].transform(
        lambda x: x.expanding().max().shift(1)
    )
    df['user_max_amount'] = df['user_max_amount'].fillna(0)
    df['exceeds_past_max'] = (
        df['amount'] > df['user_max_amount']
    ).astype(int)
    return df


def create_velocity_features(df):
    df = _sort_by_user_and_time(df)

    df['time_diff'] = (
        df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60
    )  # minutes
    df['time_diff'] = df['time_diff'].fillna(LARGE_TIME_DELTA_MINUTES)

    df = df.set_index('timestamp')
    df['transactions_last_hour'] = (
        df.groupby('user_id')['amount']
        .rolling('1h', closed='left')
        .count()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(int)
    )
    df['transactions_last_day'] = (
        df.groupby('user_id')['amount']
        .rolling('1d', closed='left')
        .count()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(int)
    )
    df = df.reset_index()
    return df


def create_category_features(df):
    df = _sort_by_user_and_time(df)

    df['category_seen_count'] = df.groupby(
        ['user_id', 'merchant_category']
    ).cumcount()
    df['is_new_category'] = (df['category_seen_count'] == 0).astype(int)

    high_risk_categories = ['electronics', 'jewelry', 'travel', 'online']
    df['is_high_risk_category'] = (
        df['merchant_category'].isin(high_risk_categories).astype(int)
    )
    
    df_temp = df[['user_id', 'timestamp', 'merchant_category']].copy()
    unique_categories = []
    for idx, row in df.iterrows():
        user_mask = df_temp['user_id'] == row['user_id']
        time_mask = (
            (df_temp['timestamp'] < row['timestamp']) &
            (df_temp['timestamp'] >= row['timestamp'] - pd.Timedelta(days=1))
        )
        cats_in_window = df_temp.loc[
            user_mask & time_mask, 'merchant_category'
        ].nunique()
        unique_categories.append(cats_in_window)

    df['unique_categories_last_day'] = unique_categories

    return df


def create_fraud_indicator_features(df):
    df = _sort_by_user_and_time(df)

    df['is_round_amount'] = (df['amount'] % 100 == 0).astype(int)
    df['is_high_value'] = (
        df['amount'] > df['amount'].quantile(0.95)
    ).astype(int)

    df = df.set_index('timestamp')
    df['amount_last_hour'] = (
        df.groupby('user_id')['amount']
        .rolling('1h', closed='left')
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    df['amount_last_day'] = (
        df.groupby('user_id')['amount']
        .rolling('1d', closed='left')
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    df = df.reset_index()

    df['amount_vs_hourly_avg'] = df['amount'] / (
        df['amount_last_hour'] / df['transactions_last_hour'].replace(0, 1) + 1
    )
    df['is_rapid_transaction'] = (
        df['time_diff'] < RAPID_TRANSACTION_THRESHOLD_MINUTES
    ).astype(int)

    return df


def create_location_features(df):
    df = _sort_by_user_and_time(df)
    
    df['location_frequency'] = df.groupby(['user_id', 'location']).cumcount()
    df['is_new_location'] = (df['location_frequency'] == 0).astype(int)

    return df


def encode_categorical(df, fit_encoder=True, encoder=None):
    df = df.copy()
    categorical_cols = ['merchant_category', 'location']

    if fit_encoder:  # training mode
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            drop=None
        )
        encoded_array = encoder.fit_transform(df[categorical_cols])
    else:
        if encoder is None:
            raise ValueError("Must provide encoder when fit_encoder=False")
        encoded_array = encoder.transform(df[categorical_cols])

    feature_names = encoder.get_feature_names_out(categorical_cols)

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=feature_names,
        index=df.index
    )

    df = pd.concat([df, encoded_df], axis=1)

    return df, encoder


def prepare_features(
    df, fit_scaler=True, scaler=None, fit_encoder=True, encoder=None
):
    logger.info("Starting feature engineering pipeline")

    logger.info("Creating time features...")
    df = create_time_features(df)

    logger.info("Creating user behavior features...")
    df = create_user_features(df)

    logger.info("Creating velocity features...")
    df = create_velocity_features(df)

    logger.info("Creating category features...")
    df = create_category_features(df)

    logger.info("Creating fraud indicators...")
    df = create_fraud_indicator_features(df)

    logger.info("Creating location features...")
    df = create_location_features(df)

    logger.info("Encoding categorical variables...")
    df, encoder = encode_categorical(
        df, fit_encoder=fit_encoder, encoder=encoder
    )

    feature_cols = [
        'amount', 'amount_deviation', 'amount_zscore',
        'amount_vs_median', 'exceeds_past_max',
        'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_business_hours',
        'hour_sin', 'hour_cos',
        'user_transaction_count', 'user_avg_amount', 'user_std_amount',
        'time_diff', 'transactions_last_hour', 'transactions_last_day',
        'is_new_category', 'is_new_location',
        'is_round_amount', 'is_high_value',
        'amount_last_hour', 'amount_last_day', 'amount_vs_hourly_avg',
        'is_rapid_transaction', 'is_high_risk_category',
        'location_frequency', 'unique_categories_last_day',
    ]

    encoded_cols = [
        col for col in df.columns
        if col.startswith(('merchant_category_', 'location_'))
    ]
    feature_cols.extend(encoded_cols)

    X = df[feature_cols].copy()
    y = df['is_fraud'].copy()

    X.fillna(0, inplace=True)

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Must provide scaler when fit_scaler=False")
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    logger.info(f"Feature engineering complete. Shape: {X.shape}")
    return X_scaled, y, feature_cols, scaler, encoder


if __name__ == '__main__':
    try:
        df = load_data()
        X, y, features, scaler, encoder = prepare_features(df)
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
    except FileNotFoundError:
        logger.error("Data file not found. Can't proceed")
        exit(1)
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        exit(1)