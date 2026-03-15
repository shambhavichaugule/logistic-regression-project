# Logistic Regression — Predicting Shipment Delays

## Project Summary

Built a logistic regression model to predict whether a shipment will be delayed using a real-world Nigerian retail and e-commerce supply chain dataset from Hugging Face. This project covers the full ML workflow — data loading, exploratory data analysis, feature engineering, model training, and evaluation.

**Dataset:** [electricsheepafrica/nigerian_retail_and_ecommerce_supply_chain_logistics_data](https://huggingface.co/datasets/electricsheepafrica/nigerian_retail_and_ecommerce_supply_chain_logistics_data)

**Business Problem:** Can we predict whether a shipment will be delayed before it ships — so logistics teams can intervene early?

---

## What I Built

A binary classification model that predicts:
- `0` → Shipment arrives on time
- `1` → Shipment is delayed (arrives more than 2 days after expected delivery date)

---

## Key Learnings

### 1. Linear Regression vs Logistic Regression

| | Linear Regression | Logistic Regression |
|---|---|---|
| Predicts | A number (e.g. price) | A category (e.g. delayed/on time) |
| Output | Any number | 0 to 1 (probability) |
| Example | What is this car's price? | Will this shipment be delayed? |

### 2. Data Quality Issues in Real Datasets

During EDA, discovered that the original `delivery_status` labels were **inconsistent with actual delivery dates**:
- 199,725 shipments marked "delivered" actually arrived late
- 8,133 shipments marked "delayed" actually arrived early

**Lesson:** Never trust dataset labels blindly. Always validate labels against raw data before modelling.

**Fix:** Created a custom target variable based on actual dates:
```python
df['target'] = (df['days_late'] > 2).astype(int)
```

### 3. Class Imbalance Problem

Original dataset distribution:
```
delivered     319,577  (91%)
delayed        32,293   (9%)
```

A model that predicts "delivered" for everything gets 91% accuracy but catches 0% of delays — completely useless in production.

**Fix:** Used `class_weight='balanced'` to force the model to pay equal attention to both classes:
```python
model = LogisticRegression(max_iter=1000, class_weight='balanced')
```

**Lesson:** Accuracy is a misleading metric on imbalanced datasets. Use recall, precision, and F1 score instead.

### 4. Data Leakage

First model achieved 100% accuracy — which was fake. The feature `days_to_deliver` contained `actual_delivery_date`, the same date used to create the target variable.

**Leaky features removed:**
```python
# These use actual_delivery_date — not available at prediction time
days_to_deliver   ❌
days_late         ❌
```

**Safe features kept:**
```python
# These are all known at the time of shipping
quantity                      ✅
shipping_cost_ngn             ✅
logistics_company             ✅
origin_city                   ✅
destination_city              ✅
shipping_duration_expected    ✅
```

**Lesson:** Always ask — "Would this information be available at the time of prediction in production?" If no, remove it.

### 5. Accuracy vs Confusion Matrix

**Accuracy** is a single number — easy to understand but misleading when classes are imbalanced.

**Confusion matrix** shows the full picture:

```
                  Predicted On Time    Predicted Delayed
Actual On Time         TN                    FP
Actual Delayed         FN                    TP
```

| Metric | What it means | When it matters |
|---|---|---|
| Accuracy | % of correct predictions overall | Balanced datasets |
| Precision | Of predicted delays, how many were real delays | When false alarms are costly |
| Recall | Of actual delays, how many did we catch | When missing a delay is costly |
| F1 Score | Balance of precision and recall | Imbalanced datasets |

For shipment delay prediction — **recall matters most**. Missing a delay is worse than a false alarm.

### 6. Feature Engineering from Dates

Raw date columns are not useful for ML. Transformed them into meaningful numeric features:

```python
df['days_to_deliver'] = (actual_delivery_date - ship_date).dt.days
df['days_late'] = (actual_delivery_date - expected_delivery_date).dt.days
df['shipping_duration_expected'] = (expected_delivery_date - ship_date).dt.days
```

### 7. Label Encoding

Machine learning models only understand numbers. Used `LabelEncoder` to convert text columns to numbers:

```python
le = LabelEncoder()
df['logistics_company_enc'] = le.fit_transform(df['logistics_company'])
df['origin_city_enc'] = le.fit_transform(df['origin_city'])
df['destination_city_enc'] = le.fit_transform(df['destination_city'])
```

### 8. Train/Test Split

Split data 80/20 to simulate real production performance:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Lesson:** Always evaluate on data the model has never seen. R² or accuracy on training data is meaningless.

---

## Final Model Performance

```
Accuracy: 0.50
Recall (delayed): 0.49
F1 Score: 0.42
```

Low performance was expected — the dataset was synthetically generated with no real relationship between features and delivery status. The value of this project was in the process, not the score.

---

## Tools & Libraries

```python
datasets          # Hugging Face dataset loading
pandas            # Data manipulation
numpy             # Numerical operations
scikit-learn      # Model training and evaluation
matplotlib        # Visualisation
python-dotenv     # Environment variable management
huggingface_hub   # Authentication
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/logistic-regression-project.git
cd logistic-regression-project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Hugging Face token to .env
echo "HF_TOKEN=your_token_here" > .env

# Run the model
python logistic_regression.py
```

---

## PM Perspective

This project simulates a real product decision: **should we build a shipment delay prediction model?**

**Before building:**
- Define what "delayed" means for the business (we chose > 2 days late)
- Validate that your labels are trustworthy (ours weren't)
- Understand class distribution before picking evaluation metrics

**In production:**
- A model predicting "on time" for everything looks great on paper but is useless
- Recall matters more than accuracy for delay detection — missing a delay costs more than a false alarm
- Data quality issues must be resolved before any model will work

**Key insight:** The most important skill in ML is not building models — it's asking the right questions about your data before you build anything.

---

## Next Steps

- Add aggregate historical features (average delivery time per logistics company and route)
- Try multinomial classification to predict all 4 delivery statuses
- Move to a cleaner dataset with real signal to see a properly performing model
- Explore Random Forest and XGBoost for comparison

---
