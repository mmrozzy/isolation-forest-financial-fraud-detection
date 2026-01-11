import pandas as pd
import numpy as np
import os

FILEPATH = os.path.join('data','transactions.csv')

def load_data(filepath=FILEPATH):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

if __name__ == '__main__':
    load_data()