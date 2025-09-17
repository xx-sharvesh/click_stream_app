import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error, silhouette_score
import joblib

# -------------------
# 1. Load train & test session-level CSVs
# -------------------
train_df = pd.read_csv("session_features.csv")           # from feature_engineering.py
test_df = pd.read_csv("session_features_test.csv")       # you said you have test_data.csv — make sure it's engineered too!

# Drop session_id because it’s just an identifier
drop_cols = ['session_id']

# -------------------
# 2. Classification (purchase_flag)
# -------------------
X_train_class = train_df.drop(columns=drop_cols + ['purchase_flag','total_spend'])
y_train_class = train_df['purchase_flag']

X_test_class = test_df.drop(columns=drop_cols + ['purchase_flag','total_spend'])
y_test_class = test_df['purchase_flag']

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_class, y_train_class)

y_pred_class = clf.predict(X_test_class)
acc = accuracy_score(y_test_class, y_pred_class)
cm = confusion_matrix(y_test_class, y_pred_class)

# -------------------
# 3. Regression (total_spend)
# -------------------
X_train_reg = train_df.drop(columns=drop_cols + ['purchase_flag','total_spend'])
y_train_reg = train_df['total_spend']

X_test_reg = test_df.drop(columns=drop_cols + ['purchase_flag','total_spend'])
y_test_reg = test_df['total_spend']

regr = RandomForestRegressor(n_estimators=200, random_state=42)
regr.fit(X_train_reg, y_train_reg)

y_pred_reg = regr.predict(X_test_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)

# -------------------
# 4. Clustering (KMeans)
# -------------------
X_cluster = pd.concat([train_df, test_df], axis=0)  # cluster on all sessions
X_cluster_features = X_cluster.drop(columns=drop_cols + ['purchase_flag','total_spend'])

# scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster_features)

# choose k=3 (example); you can adjust after elbow analysis
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

labels = kmeans.predict(scaler.transform(X_cluster_features))
sil_score = silhouette_score(X_scaled, labels)

# -------------------
# 5. Save metrics
# -------------------
metrics = {
    'classification_accuracy': acc,
    'classification_confusion_matrix': cm.tolist(),
    'regression_r2': r2,
    'regression_mae': mae,
    'clustering_silhouette': sil_score
}

print("✅ Metrics:\n", metrics)

# -------------------
# 6. Save models
# -------------------
joblib.dump(clf, "rf_classifier.pkl")
joblib.dump(regr, "rf_regressor.pkl")
joblib.dump((scaler, kmeans), "kmeans_model.pkl")  # store scaler & model together

# also save metrics as JSON
import json
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Models saved. Classifier: rf_classifier.pkl, Regressor: rf_regressor.pkl, Clustering: kmeans_model.pkl")
