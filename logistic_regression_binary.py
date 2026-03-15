import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = load_dataset("electricsheepafrica/nigerian_retail_and_ecommerce_supply_chain_logistics_data")
df = dataset['train'].to_pandas()

# Check delivery_status values
# print(df['delivery_status'].value_counts())

# Step 1 — Keep only delivered and delayed
# df = df[df['delivery_status'].isin(['delivered', 'delayed'])]

# Step 2 — Convert target to binary
# df['target'] = (df['delivery_status'] == 'delivered').astype(int)
# print("\nTarget distribution:")
# print(df['target'].value_counts())

# Step 3 — Convert text columns to numbers using Label Encoding
le = LabelEncoder()
df['logistics_company_enc'] = le.fit_transform(df['logistics_company'])
df['origin_city_enc'] = le.fit_transform(df['origin_city'])
df['destination_city_enc'] = le.fit_transform(df['destination_city'])

# Convert date columns to datetime
df['ship_date'] = pd.to_datetime(df['ship_date'])
df['expected_delivery_date'] = pd.to_datetime(df['expected_delivery_date'])
df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'])

# Create new features from dates
df['days_to_deliver'] = (df['actual_delivery_date'] - df['ship_date']).dt.days
df['days_late'] = (df['actual_delivery_date'] - df['expected_delivery_date']).dt.days
df['shipping_duration_expected'] = (df['expected_delivery_date'] - df['ship_date']).dt.days

# print(df[['days_to_deliver', 'days_late', 'shipping_duration_expected', 'delivery_status']].head(10))
# print("\nDays late stats:")
# print(df['days_late'].describe())

# Compare days_late for delayed vs delivered
# print(df.groupby('delivery_status')['days_late'].describe())

# How many delayed shipments actually arrived early?
# delayed_but_early = df[(df['delivery_status'] == 'delayed') & (df['days_late'] < 0)]
# print(f"\nDelayed but arrived early: {len(delayed_but_early)}")

# How many delivered shipments actually arrived late?
# delivered_but_late = df[(df['delivery_status'] == 'delivered') & (df['days_late'] > 0)]
# print(f"Delivered but arrived late: {len(delivered_but_late)}")

# Based on the above analysis, we can set a threshold for days_late to define our target variable.
df['target'] = (df['days_late'] > 2).astype(int)
# print("Target distribution:")
# print(df['target'].value_counts())
# print("\nPercentage:")
# print(df['target'].value_counts(normalize=True) * 100)

# Step 4 — Select features
X = df[['quantity', 'shipping_cost_ngn', 'logistics_company_enc', 
        'origin_city_enc', 'destination_city_enc', 'shipping_duration_expected']]
y = df['target']

# Step 5 — Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6 — Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Step 7 — Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Step 9 — Save chart
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0, 1], ['On Time', 'Delayed'])
plt.yticks([0, 1], ['On Time', 'Delayed'])
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nChart saved as confusion_matrix.png")
