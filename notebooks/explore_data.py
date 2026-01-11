import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/transactions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Dataset Overview:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

print("\nFaud vs Normal Comparison")
print(df.groupby('is_fraud')['amount'].describe())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Amount distribution
axes[0, 0].hist(df[df['is_fraud']==0]['amount'], bins=50, alpha=0.7, label='Normal', color='green')
axes[0, 0].hist(df[df['is_fraud']==1]['amount'], bins=50, alpha=0.7, label='Fraud', color='red')
axes[0, 0].set_xlabel('Transaction Amount')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Transaction Amount Distribution')
axes[0, 0].legend()

# Transactions by hour
df['hour'] = df['timestamp'].dt.hour
hour_fraud = df[df['is_fraud']==1].groupby('hour').size()
hour_normal = df[df['is_fraud']==0].groupby('hour').size()

axes[0, 1].plot(hour_normal.index, hour_normal.values, label='Normal', color='green', marker='o')
axes[0, 1].plot(hour_fraud.index, hour_fraud.values, label='Fraud', color='red', marker='o')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Number of Transactions')
axes[0, 1].set_title('Transactions by Hour')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Category distribution
category_counts = df.groupby(['merchant_category', 'is_fraud']).size().unstack(fill_value=0)
category_counts.plot(kind='bar', ax=axes[1, 0], color=['green', 'red'])
axes[1, 0].set_xlabel('Merchant Category')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Transactions by Category')
axes[1, 0].legend(['Normal', 'Fraud'])
axes[1, 0].tick_params(axis='x', rotation=45)

# Fraud rate by city
city_fraud = df.groupby('location')['is_fraud'].agg(['sum', 'count'])
city_fraud['rate'] = city_fraud['sum'] / city_fraud['count'] * 100
city_fraud['rate'].plot(kind='bar', ax=axes[1, 1], color='orange')
axes[1, 1].set_xlabel('City')
axes[1, 1].set_ylabel('Fraud Rate (%)')
axes[1, 1].set_title('Fraud Rate by Location')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('data/exploratory_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: data/exploratory_analysis.png")
plt.show()