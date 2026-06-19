# ============================================================
# Smart Diabetes Risk Prediction System
# main.py — Data Processing, EDA, Model Training & Saving
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving images
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

# ─── Paths ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'dataset', 'diabetes.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
IMAGE_DIR  = os.path.join(BASE_DIR, 'images')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
# PHASE 2 — Load Dataset
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SMART DIABETES RISK PREDICTION SYSTEM")
print("="*60)

print("\n[Phase 2] Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

# ════════════════════════════════════════════════════════════
# PHASE 3 — Data Understanding
# ════════════════════════════════════════════════════════════
print("\n[Phase 3] Data Understanding")
print("\n--- First 5 rows ---")
print(df.head())
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Statistical Summary ---")
print(df.describe())
print("\n--- Target Distribution ---")
print(df['Outcome'].value_counts())

# ════════════════════════════════════════════════════════════
# PHASE 4 — Data Preprocessing
# ════════════════════════════════════════════════════════════
print("\n[Phase 4] Data Preprocessing …")

# Check missing values
print("\n  Missing values:")
print(df.isnull().sum())

# Remove duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n  Duplicates removed: {before - len(df)}")

# Replace invalid 0 values with column median
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    median_val = df[col].median()
    df[col] = df[col].replace(0, median_val)
    print(f"  Replaced 0s in '{col}' with median ({median_val:.2f})")

print("\n  Preprocessing complete ✓")

# ════════════════════════════════════════════════════════════
# PHASE 5 — Exploratory Data Analysis (EDA)
# ════════════════════════════════════════════════════════════
print("\n[Phase 5] Performing EDA & saving visualisations …")

sns.set_style("whitegrid")
PALETTE = ["#4C9BE8", "#E8544C"]

# --- 1. Outcome Distribution ---
fig, ax = plt.subplots(figsize=(6, 4))
outcome_counts = df['Outcome'].value_counts()
colors = ["#4C9BE8", "#E8544C"]
bars = ax.bar(['Non-Diabetic (0)', 'Diabetic (1)'],
              outcome_counts.values, color=colors, width=0.5, edgecolor='white')
for bar, val in zip(bars, outcome_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha='center', fontsize=11, fontweight='bold')
ax.set_title('Outcome Distribution', fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Count')
ax.set_ylim(0, outcome_counts.max() + 40)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'outcome_distribution.png'), dpi=120)
plt.close()

# --- 2. Feature Histograms ---
features = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, feat in enumerate(features):
    axes[i].hist(df[feat], bins=20, color='#4C9BE8', edgecolor='white', alpha=0.85)
    axes[i].set_title(feat, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'feature_histograms.png'), dpi=120, bbox_inches='tight')
plt.close()

# --- 3. Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            mask=mask, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'correlation_heatmap.png'), dpi=120)
plt.close()

# --- 4. Glucose by Outcome ---
fig, ax = plt.subplots(figsize=(7, 4))
for outcome, color, label in zip([0, 1], PALETTE, ['Non-Diabetic', 'Diabetic']):
    data = df[df['Outcome'] == outcome]['Glucose']
    ax.hist(data, bins=20, color=color, alpha=0.7, label=label, edgecolor='white')
ax.set_title('Glucose Distribution by Outcome', fontsize=13, fontweight='bold')
ax.set_xlabel('Glucose')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'glucose_analysis.png'), dpi=120)
plt.close()

# --- 5. BMI by Outcome ---
fig, ax = plt.subplots(figsize=(7, 4))
df.boxplot(column='BMI', by='Outcome', ax=ax,
           boxprops=dict(color='#4C9BE8'),
           medianprops=dict(color='#E8544C', linewidth=2),
           whiskerprops=dict(color='#4C9BE8'),
           capprops=dict(color='#4C9BE8'))
ax.set_title('BMI Distribution by Outcome', fontsize=13, fontweight='bold')
ax.set_xlabel('Outcome (0 = Non-Diabetic, 1 = Diabetic)')
ax.set_ylabel('BMI')
plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'bmi_analysis.png'), dpi=120)
plt.close()

# --- 6. Age vs Glucose Scatter ---
fig, ax = plt.subplots(figsize=(7, 5))
for outcome, color, label in zip([0, 1], PALETTE, ['Non-Diabetic', 'Diabetic']):
    sub = df[df['Outcome'] == outcome]
    ax.scatter(sub['Age'], sub['Glucose'], c=color, alpha=0.55, label=label, s=25)
ax.set_title('Age vs Glucose by Outcome', fontsize=13, fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Glucose')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'age_glucose_scatter.png'), dpi=120)
plt.close()

print("  EDA charts saved to /images/ ✓")

# ════════════════════════════════════════════════════════════
# PHASE 6 & 7 — Feature Selection & Train-Test Split
# ════════════════════════════════════════════════════════════
print("\n[Phase 6 & 7] Feature Selection & Train-Test Split …")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y)

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")

# Save scaler
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print("  Scaler saved ✓")

# ════════════════════════════════════════════════════════════
# PHASE 8 — Model Training
# ════════════════════════════════════════════════════════════
print("\n[Phase 8] Training ML Models …")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'model'    : model,
        'y_pred'   : y_pred,
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'f1'       : f1_score(y_test, y_pred)
    }
    print(f"\n  {name}")
    print(f"    Accuracy : {results[name]['accuracy']:.4f}")
    print(f"    Precision: {results[name]['precision']:.4f}")
    print(f"    Recall   : {results[name]['recall']:.4f}")
    print(f"    F1-Score : {results[name]['f1']:.4f}")

# ════════════════════════════════════════════════════════════
# PHASE 9 — Model Evaluation Charts
# ════════════════════════════════════════════════════════════
print("\n[Phase 9] Saving evaluation charts …")

# --- Model Comparison Bar Chart ---
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
model_names  = list(results.keys())
x = np.arange(len(metric_names))
width = 0.25
colors_bar = ['#4C9BE8', '#E8A84C', '#4CE87A']

fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, color) in enumerate(zip(model_names, colors_bar)):
    vals = [results[name]['accuracy'], results[name]['precision'],
            results[name]['recall'],  results[name]['f1']]
    bars = ax.bar(x + i*width, vals, width, label=name, color=color,
                  edgecolor='white', alpha=0.88)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=10)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.12)
ax.set_xticks(x + width)
ax.set_xticklabels(metric_names)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'model_comparison.png'), dpi=120)
plt.close()

# --- Confusion Matrix for Random Forest ---
rf_pred = results['Random Forest']['y_pred']
cm = confusion_matrix(y_test, rf_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'],
            linewidths=1, ax=ax)
ax.set_title('Random Forest — Confusion Matrix', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'confusion_matrix.png'), dpi=120)
plt.close()

# --- Feature Importance ---
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
colors_fi = ['#4C9BE8' if v < importances.max() else '#E8544C' for v in importances.values]
importances.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'feature_importance.png'), dpi=120)
plt.close()

print("  Evaluation charts saved ✓")

# ════════════════════════════════════════════════════════════
# PHASE 10 — Save Best Model (Random Forest)
# ════════════════════════════════════════════════════════════
print("\n[Phase 10] Saving best model (Random Forest) …")
best_model = results['Random Forest']['model']
with open(os.path.join(MODEL_DIR, 'diabetes_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

# Save model metrics for dashboard
metrics_data = {name: {k: v for k, v in data.items() if k != 'model' and k != 'y_pred'}
                for name, data in results.items()}
# Save accuracy summary
with open(os.path.join(MODEL_DIR, 'model_metrics.pkl'), 'wb') as f:
    pickle.dump(metrics_data, f)

print(f"  diabetes_model.pkl saved ✓")
print(f"  model_metrics.pkl  saved ✓")

# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  TRAINING COMPLETE!")
print("="*60)
rf_acc = results['Random Forest']['accuracy']
print(f"\n  Best Model    : Random Forest Classifier")
print(f"  Accuracy      : {rf_acc*100:.2f}%")
print(f"  Saved to      : models/diabetes_model.pkl")
print(f"\n  Run the dashboard with:  streamlit run app.py")
print("="*60 + "\n")
