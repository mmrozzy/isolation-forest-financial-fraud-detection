from faker import Faker
import numpy as np
import random
from datetime import datetime, timedelta
import pandas as pd
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

NUM_USERS = 150
NUM_TRANSACTIONS = 12000
FRAUD_RATE = 0.025  # 2.5% fraud rate
START_DATE = datetime(2024, 6, 1)
END_DATE = datetime(2024, 12, 31)

MERCHANT_CATEGORIES = {
    'grocery': (15, 150),
    'gas': (30, 80),
    'restaurant': (20, 100),
    'shopping': (25, 200),
    'entertainment': (15, 80),
    'utilities': (50, 300),
    'healthcare': (40, 250),
    'travel': (100, 800),
    'electronics': (50, 1200),
    'jewelry': (200, 3000)
}

CITIES = ['Montreal', 'Toronto', 'Vancouver', 'Ottawa', 'Calgary', 
          'Edmonton', 'Quebec City', 'Winnipeg', 'Halifax', 'Victoria']

FRAUD_TYPES = ['high_value_unusual_time', 'geographic_anomaly', 
                   'unusual_category', 'high_velocity', 'excessive_amount']

def generate_timestamp(start, end):
    delta = end - start
    random_second = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_second)

def generate_normal_transaction(user_id, user_profile):
    timestamp = generate_timestamp(START_DATE, END_DATE)
    hour = random.randint(6,23)
    timestamp = timestamp.replace(hour=hour, minute=random.randint(0,59))

    category = random.choice(user_profile['preferred_categories'])
    amount_range = MERCHANT_CATEGORIES[category]

    amount = np.random.normal(
        loc=user_profile['avg_amount'],
        scale=user_profile['avg_amount'] * 0.3
    )
    amount = np.clip(amount, amount_range[0], amount_range[1])
    amount = round(amount,2)

    return {
        'user_id': user_id,
        'timestamp': timestamp,
        'amount': amount,
        'merchant_category': category,
        'merchant_name': fake.company(),
        'location': user_profile['home_city'],
        'is_fraud': 0
    }

def generate_fraud_transaction(user_id, user_profile, fraud_type):

    if fraud_type == 'high_value_unusual_time':

        timestamp = generate_timestamp(START_DATE, END_DATE)
        hour = random.randint(2, 5)
        timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))
        
        amount = random.uniform(800, 3500)
        category = random.choice(['electronics', 'jewelry', 'travel'])
        location = user_profile['home_city']
    
    elif fraud_type == 'geographic_anomaly':
        timestamp = generate_timestamp(START_DATE, END_DATE)
        hour = random.randint(0, 23)
        timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))
        
        different_cities = [c for c in CITIES if c != user_profile['home_city']]
        location = random.choice(different_cities)
        
        amount = random.uniform(300, 1500)
        category = random.choice(['electronics', 'jewelry', 'shopping'])
        
    elif fraud_type == 'unusual_category':
        timestamp = generate_timestamp(START_DATE, END_DATE)
        hour = random.randint(0, 23)
        timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))
        
        unusual_categories = ['jewelry', 'electronics', 'travel']
        unusual_categories = [c for c in unusual_categories 
                             if c not in user_profile['preferred_categories']]
        category = random.choice(unusual_categories) if unusual_categories else 'jewelry'
        
        amount = random.uniform(500, 2500)
        location = user_profile['home_city']
        
    elif fraud_type == 'high_velocity':
        timestamp = generate_timestamp(START_DATE, END_DATE)
        timestamp += timedelta(minutes=random.randint(0, 15))
        
        amount = random.uniform(100, 800)
        category = random.choice(list(MERCHANT_CATEGORIES.keys()))
        location = user_profile['home_city']
    
    else: # excessive_amount
        timestamp = generate_timestamp(START_DATE, END_DATE)
        hour = random.randint(0, 23)
        timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))
        
        amount = user_profile['avg_amount'] * random.uniform(8, 15)
        category = random.choice(['jewelry', 'electronics'])
        location = user_profile['home_city']
    
    return {
        'user_id': user_id,
        'timestamp': timestamp,
        'amount': round(amount, 2),
        'merchant_category': category,
        'merchant_name': fake.company(),
        'location': location,
        'is_fraud': 1
    }

def create_user_profiles(num_users):
    profiles = {}

    for user_id in range(1, num_users + 1):

        # categories
        num_preferred = random.randint(3,6)
        preferred_categories = random.sample(list(MERCHANT_CATEGORIES.keys()), num_preferred)

        # spending
        avg_amount = random.uniform(50,250)

        profiles[user_id] = {
            'home_city' : random.choice(CITIES),
            'preferred_categories' : preferred_categories,
            'avg_amount' : avg_amount
        }
    
    return profiles

def generate_dataset():
    user_profiles = create_user_profiles(NUM_USERS)

    num_fraud = int(NUM_TRANSACTIONS * FRAUD_RATE)
    num_normal = NUM_TRANSACTIONS - num_fraud

    transactions = []

    print(f"Generating {num_normal} normal transactions ...")
    for _ in range(num_normal):
        user_id = random.randint(1, NUM_USERS)
        transaction = generate_normal_transaction(user_id, user_profiles[user_id])
        transactions.append(transaction)

    print(f"Generating {num_fraud} fraudulent transactions ...")
    for _ in range(num_fraud):
        user_id = random.randint(1, NUM_USERS)
        fraud_type = random.choice(FRAUD_TYPES)
        transaction = generate_fraud_transaction(user_id, user_profiles[user_id], fraud_type)
        transactions.append(transaction)

    df = pd.DataFrame(transactions)
    df = df.sort_values('timestamp').reset_index(drop=True)
    print("Generating transaction ids ...")
    df.insert(0, 'transaction_id', range(1, len(df) + 1))  

    print("Data generation completed.")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


if __name__ == '__main__':
    
    df = generate_dataset()

    output_path = os.path.join('data','transactions.csv')
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
