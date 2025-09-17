import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# 1. Load the data
# -------------------
df = pd.read_csv("train_data.csv")

# Create output folder for plots
os.makedirs("eda_plots", exist_ok=True)

# -------------------
# 2. Basic data info
# -------------------
print("Shape of dataset:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# -------------------
# 3. Handle missing values (basic)
# -------------------
# Numeric columns: fill with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical columns: fill with mode
cat_cols = df.select_dtypes(exclude=[np.number]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------
# 4. Visualization helper
# -------------------
def save_plot(fig, name):
    fig.savefig(f"eda_plots/{name}.png", bbox_inches="tight")
    plt.close(fig)

# -------------------
# 5. Visualizations
# -------------------

# 5.1 Distribution of sessions per country
fig, ax = plt.subplots(figsize=(12,6))
top_countries = df['country'].value_counts().head(15)
sns.barplot(x=top_countries.index, y=top_countries.values, ax=ax)
ax.set_title("Top 15 Countries by Session Count")
ax.set_xlabel("Country")
ax.set_ylabel("Number of Sessions")
plt.xticks(rotation=45)
save_plot(fig, "sessions_per_country")

# 5.2 Main product category popularity
fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(x='page1_main_category', data=df, ax=ax)
ax.set_title("Main Product Category Distribution")
ax.set_xlabel("Main Category")
plt.xticks(rotation=0)
save_plot(fig, "main_category_distribution")

# 5.3 Color preferences
fig, ax = plt.subplots(figsize=(10,6))
top_colors = df['colour'].value_counts().head(10)
sns.barplot(x=top_colors.index, y=top_colors.values, ax=ax)
ax.set_title("Top 10 Color Preferences")
ax.set_xlabel("Color")
ax.set_ylabel("Count")
plt.xticks(rotation=45)
save_plot(fig, "color_preferences")

# 5.4 Price distribution
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(df['price'], bins=30, kde=True, ax=ax)
ax.set_title("Price Distribution")
ax.set_xlabel("Price (USD)")
save_plot(fig, "price_distribution")

# 5.5 Session length (clicks per session)
if 'session_id' in df.columns and 'order' in df.columns:
    session_counts = df.groupby('session_id')['order'].count()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(session_counts, bins=30, kde=False, ax=ax)
    ax.set_title("Number of Clicks per Session")
    ax.set_xlabel("Clicks per Session")
    save_plot(fig, "session_length_distribution")

# 5.6 Correlation heatmap (numeric features)
if len(numeric_cols) > 1:
    fig, ax = plt.subplots(figsize=(10,8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    save_plot(fig, "correlation_heatmap")

# 5.7 Time-based analysis (by month)
if 'month' in df.columns:
    fig, ax = plt.subplots(figsize=(8,6))
    sns.countplot(x='month', data=df, ax=ax)
    ax.set_title("Session Count by Month")
    ax.set_xlabel("Month")
    save_plot(fig, "sessions_by_month")

print("✅ EDA plots saved in 'eda_plots/' folder.")
