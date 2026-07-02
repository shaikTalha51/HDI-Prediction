"""
train_model.py
---------------
Epic 2: Importing Required Libraries
Epic 3: Dataset Download and Understanding
Epic 4: Data Preprocessing and Label Encoding
Epic 5: Dividing the Model into Train and Test Data
Epic 6: Fitting the Model
Epic 7: Saving the Model

Run this once to (re)train the HDI prediction model. It produces:
    - models/hdi_model.pkl   (trained LinearRegression model)
    - models/scaler.pkl      (StandardScaler fit on the training features)
    - static/eda_*.png       (exploratory data-analysis charts used by the app)
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Epic 3: Dataset Download and Understanding
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "hdi_dataset.csv")
print("Shape:", df.shape)
print(df.describe())

FEATURES = [
    "Life_Expectancy",
    "Mean_Years_Schooling",
    "Expected_Years_Schooling",
    "GNI_Per_Capita",
]
TARGET = "HDI_Score"


def classify_hdi(score: float) -> str:
    """UNDP-style HDI tier classification."""
    if score >= 0.80:
        return "Very High"
    elif score >= 0.70:
        return "High"
    elif score >= 0.55:
        return "Medium"
    return "Low"


df["HDI_Tier"] = df[TARGET].apply(classify_hdi)

# ---------------------------------------------------------------------
# Epic 4: Data Preprocessing and Label Encoding
# ---------------------------------------------------------------------
# Fill missing numeric values with column mean
for col in FEATURES:
    df[col] = df[col].fillna(df[col].mean())

print("\nMissing values after cleaning:\n", df.isnull().sum())

X = df[FEATURES]
y = df[TARGET]

# ---------------------------------------------------------------------
# Data Visualization & Analysis (saved for the Flask "Insights" page)
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 6))
sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(STATIC_DIR / "eda_heatmap.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="GNI_Per_Capita", y=TARGET, hue="HDI_Tier", palette="viridis")
plt.title("GNI per Capita vs HDI Score")
plt.tight_layout()
plt.savefig(STATIC_DIR / "eda_scatter.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 5))
sns.stripplot(data=df, x="HDI_Tier", y="Life_Expectancy",
              order=["Low", "Medium", "High", "Very High"], palette="magma")
plt.title("Life Expectancy by HDI Tier")
plt.tight_layout()
plt.savefig(STATIC_DIR / "eda_strip.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 5))
sns.histplot(df[TARGET], kde=True, color="teal")
plt.title("Distribution of HDI Scores")
plt.tight_layout()
plt.savefig(STATIC_DIR / "eda_dist.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# Epic 5: Dividing the Model into Train and Test Data
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# Epic 6: Fitting the Model
# ---------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Performance:")
print(f"  R-squared : {r2:.4f}")
print(f"  MAE       : {mae:.4f}")
print(f"  RMSE      : {rmse:.4f}")
print(f"  Coefficients: {dict(zip(FEATURES, model.coef_))}")
print(f"  Intercept : {model.intercept_:.4f}")

# ---------------------------------------------------------------------
# Epic 7: Saving the Model
# ---------------------------------------------------------------------
with open(MODELS_DIR / "hdi_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(MODELS_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

metrics = {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)}
with open(MODELS_DIR / "metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print(f"\nSaved model artifacts in {MODELS_DIR} and charts in {STATIC_DIR}")
