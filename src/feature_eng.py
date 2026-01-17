import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

FILEPATH = os.path.join('data','transactions.csv')
SCALER_PATH = os.path.join('models', 'scaler.pkl')


def load_data(filepath=FILEPATH):
    df = pd.read_csv(filepath)
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
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int) #can be customized later
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

    # cyclical hours
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df

def create_user_features(df):
    df = df.copy()
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    df['user_transaction_count'] = df.groupby('user_id').cumcount()
    df['user_avg_amount'] = df.groupby('user_id')['amount'].transform( lambda x: x.expanding().mean().shift(1)) #shift to lose current entry (previous for comparison)
    df['user_std_amount'] = df.groupby('user_id')['amount'].transform( lambda x: x.expanding().std().shift(1) ) 
    # in case of no transaction history - baseline:
    df['user_avg_amount'] = df['user_avg_amount'].fillna(df['amount'].mean())
    df['user_std_amount'] = df['user_std_amount'].fillna(df['amount'].std())
    #metrics
    df['amount_deviation'] = (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1)
    df['amount_zscore'] = np.abs((df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1))
    
    df['user_median_amount'] = df.groupby('user_id')['amount'].transform(lambda x: x.expanding().median().shift(1))
    df['user_median_amount'] = df['user_median_amount'].fillna(df['amount'].median())
    df['amount_vs_median'] = df['amount'] / (df['user_median_amount'] + 1)
    
    df['user_max_amount'] = df.groupby('user_id')['amount'].transform(lambda x: x.expanding().max().shift(1))
    df['user_max_amount'] = df['user_max_amount'].fillna(0)
    df['exceeds_past_max'] = (df['amount'] > df['user_max_amount']).astype(int)

    return df

def create_velocity_features(df):
    df = df.copy()
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    df['time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60  #mins
    df['time_diff'] = df['time_diff'].fillna(10000)

    df['transactions_last_hour'] = 0
    df['transactions_last_day'] = 0

    for idn in range(len(df)):
        user = df.loc[idn, 'user_id']
        current_time = df.loc[idn, 'timestamp']

        user_history = df[(df['user_id'] == user) & (df['timestamp'] < current_time)]

        one_hour_ago = current_time - pd.Timedelta(hours=1)
        df.loc[idn, 'transactions_last_hour'] = len(user_history[user_history['timestamp'] > one_hour_ago])
        one_day_ago = current_time - pd.Timedelta(days=1)
        df.loc[idn, 'transactions_last_day'] = len(user_history[user_history['timestamp'] > one_day_ago])
    
    return df

def create_category_features(df):
    df = df.copy()
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    df['user_common_category'] = 'unknown'
    
    for idx in range(len(df)):
        user = df.loc[idx, 'user_id']
        current_time = df.loc[idx, 'timestamp']
        
        user_history = df[(df['user_id'] == user) & (df['timestamp'] < current_time)]
        
        if len(user_history) > 0:
            most_common = user_history['merchant_category'].mode()
            if len(most_common) > 0:
                df.loc[idx, 'user_common_category'] = most_common[0]

    df['is_new_category'] = 0
    for idn in range(len(df)):
        user = df.loc[idn, 'user_id']
        current_category = df.loc[idn, 'merchant_category']
        current_time = df.loc[idn, 'timestamp']

        user_history = df[(df['user_id'] == user) & (df['timestamp'] < current_time)]
        past_categories = user_history['merchant_category'].unique()

        if current_category not in past_categories and len(past_categories) > 0:
            df.loc[idn, 'is_new_category'] = 1
    
    return df

def create_fraud_indicators(df):
    """High-signal features specifically for fraud detection"""
    df = df.copy()
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    df['is_round_amount'] = (df['amount'] % 100 == 0).astype(int)
    df['is_high_value'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    
    df['amount_last_hour'] = 0
    df['amount_last_day'] = 0
    
    for idx in range(len(df)):
        user = df.loc[idx, 'user_id']
        current_time = df.loc[idx, 'timestamp']
        user_history = df[(df['user_id'] == user) & (df['timestamp'] < current_time)]
        
        one_hour_ago = current_time - pd.Timedelta(hours=1)
        df.loc[idx, 'amount_last_hour'] = user_history[user_history['timestamp'] > one_hour_ago]['amount'].sum()
        
        one_day_ago = current_time - pd.Timedelta(days=1)
        df.loc[idx, 'amount_last_day'] = user_history[user_history['timestamp'] > one_day_ago]['amount'].sum()
    
    df['amount_vs_hourly_avg'] = df['amount'] / (df['amount_last_hour'] / df['transactions_last_hour'].replace(0, 1) + 1)
    df['is_rapid_transaction'] = (df['time_diff'] < 5).astype(int) 
    
    return df

def create_location_features(df):
    """Location-based anomaly detection"""
    df = df.copy()
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    df['is_new_location'] = 0
    df['location_frequency'] = 0
    
    for idx in range(len(df)):
        user = df.loc[idx, 'user_id']
        current_location = df.loc[idx, 'location']
        current_time = df.loc[idx, 'timestamp']
        
        user_history = df[(df['user_id'] == user) & (df['timestamp'] < current_time)]
        
        if len(user_history) > 0:
            past_locations = user_history['location'].unique()
            
            if current_location not in past_locations:
                df.loc[idx, 'is_new_location'] = 1            
            if current_location in past_locations:
                df.loc[idx, 'location_frequency'] = (user_history['location'] == current_location).sum()
    
    return df

def create_category_risk_features(df):
    """Category-based risk indicators"""
    df = df.copy()
    
    high_risk_categories = ['electronics', 'jewelry', 'travel', 'online']
    df['is_high_risk_category'] = df['merchant_category'].isin(high_risk_categories).astype(int)
    
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    df['unique_categories_last_day'] = 0
    
    for idx in range(len(df)):
        user = df.loc[idx, 'user_id']
        current_time = df.loc[idx, 'timestamp']
        
        user_history = df[(df['user_id'] == user) & (df['timestamp'] < current_time)]
        one_day_ago = current_time - pd.Timedelta(days=1)
        recent_categories = user_history[user_history['timestamp'] > one_day_ago]['merchant_category'].nunique()
        
        df.loc[idx, 'unique_categories_last_day'] = recent_categories
    
    return df

def encode_categorical(df, fit_encoder=True, encoder=None):
    df = df.copy()
    categorical_cols = ['merchant_category', 'location']
    
    if fit_encoder: #training mode
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

def prepare_features(df, fit_scaler=True, scaler=None, fit_encoder=True, encoder=None):
    print("Creating time features...")
    df = create_time_features(df)
    print("Creating user behavior features...")
    df = create_user_features(df)
    print("Creating velocity features...")
    df = create_velocity_features(df)
    print("Creating category features...")
    df = create_category_features(df)
    print("Creating fraud indicators...")
    df = create_fraud_indicators(df)
    print("Creating location features...")
    df = create_location_features(df)
    print("Creating category risk features...")
    df = create_category_risk_features(df)
    print("Encoding categorical variables...")
    df, encoder = encode_categorical(df, fit_encoder=fit_encoder, encoder=encoder)
    
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
    
    encoded_cols = [col for col in df.columns if col.startswith(('merchant_category_', 'location_'))]
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
    
    return X_scaled, y, feature_cols, scaler, encoder

if __name__ == '__main__':
    df = load_data()
    X, y, features, scaler, encoder = prepare_features(df)
    print(X.head())
    joblib.dump(scaler, SCALER_PATH)
