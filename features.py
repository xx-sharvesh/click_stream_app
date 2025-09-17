import pandas as pd
import numpy as np

# -------------------
# 1. Load the data
# -------------------
df = pd.read_csv("train_data.csv")

# Make sure all column names are lower-case and no spaces
df.columns = [c.lower().strip() for c in df.columns]

# -------------------
# 2. Add time features
# -------------------
# Combine year/month/day into a datetime for each record
df['date'] = pd.to_datetime(dict(year=df['year'],
                                 month=df['month'],
                                 day=df['day']),
                            errors='coerce')

df['day_of_week'] = df['date'].dt.dayofweek   # Monday=0
df['month_name'] = df['date'].dt.month_name()

# -------------------
# 3. Purchase flag & spend
# -------------------
# Assuming "order" > 0 means purchased
df['purchase_flag'] = np.where(df['order'] > 0, 1, 0)

# Total spend in that record
df['line_spend'] = df['price'] * df['order']

# -------------------
# 4. Aggregate to session level
# -------------------
session_features = df.groupby('session_id').agg(
    n_unique_categories=('page1_main_category', 'nunique'),
    n_unique_models=('page2_clothing_model', 'nunique'),
    n_clicks=('page', 'count'),                 # number of clicks/rows
    avg_price_viewed=('price', 'mean'),
    total_price_viewed=('price', 'sum'),
    total_spend=('line_spend', 'sum'),
    purchase_flag=('purchase_flag', 'max'),     # 1 if any purchase in session
    country=('country', lambda x: x.mode()[0] if len(x)>0 else np.nan),
    most_common_category=('page1_main_category', lambda x: x.mode()[0] if len(x)>0 else np.nan),
    day_of_week=('day_of_week', lambda x: x.mode()[0] if len(x)>0 else np.nan),
    month=('month', lambda x: x.mode()[0] if len(x)>0 else np.nan)
).reset_index()

# -------------------
# 5. Optional: derive ratios
# -------------------
session_features['avg_spend_per_click'] = session_features['total_spend'] / session_features['n_clicks']
session_features['avg_price_per_click'] = session_features['total_price_viewed'] / session_features['n_clicks']

# Replace inf/nan after division
session_features = session_features.replace([np.inf, -np.inf], 0).fillna(0)

# -------------------
# 6. Save to CSV
# -------------------
session_features.to_csv("session_features.csv", index=False)
print("✅ session_features.csv created with shape:", session_features.shape)
print("Columns:\n", session_features.columns.tolist())
