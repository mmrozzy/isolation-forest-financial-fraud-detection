import pandas as pd
import numpy as np
import os

FILEPATH = os.path.join('data','transactions.csv')

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
    df['user_avg_amount'].fillna(df['amount'].mean(), inplace=True)
    df['user_std_amount'].fillna(df['amount'].std(), inplace=True)
    #metrics
    df['amount_deviation'] = (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1)
    df['amount_zscore'] = np.abs((df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1))

    return df

def create_velocity_features(df):
    df = df.copy()
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    df['time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60  #mins
    df['time_diff'].fillna(10000, inplace=True)

    df['transactions_last_hour'] = 0
    df['transactions_last_day'] = 0

    for idn in range(len(df)):
        user = df.loc[idn, 'user_id']
        current_time = df.loc[idn, 'timestamp']

        user_history = df[(df['user_id'] == user) & (df['timestamp'] < current_time)]

        # last hour
        one_hour_ago = current_time - pd.Timedelta(hours=1)
        df.loc[idn, 'transactions_last_hour'] = len(user_history[user_history['timestamp'] > one_hour_ago])
        # last day 
        one_day_ago = current_time - pd.Timedelta(days=1)
        df.loc[idn, 'transactions_last_day'] = len(user_history[user_history['timestamp'] > one_day_ago])
    
    return df

# def create_category_features(df):
#     df = df.copy
#     df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)




if __name__ == '__main__':
    load_data()