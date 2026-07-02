"""
generate_dataset.py
--------------------
Generates a realistic synthetic HDI dataset (since live Kaggle download
requires internet access that isn't available in this environment).

The dataset mimics the real UNDP Human Development Report structure:
    Country, Life Expectancy, Mean Years of Schooling,
    Expected Years of Schooling, GNI per Capita (PPP $), HDI Score

Replace this step in your own environment with a direct Kaggle download,
e.g.:
    import kagglehub
    path = kagglehub.dataset_download("<dataset-owner>/<dataset-name>")
"""

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

N = 200

countries = [f"Country_{i+1}" for i in range(N)]

# Simulate four human-development "clusters" so the data has realistic
# spread across Low / Medium / High / Very High tiers.
tier_sizes = [40, 60, 60, 40]  # Low, Medium, High, Very High
tier_names = ["Low", "Medium", "High", "Very High"]

rows = []
idx = 0
for tier, size in zip(tier_names, tier_sizes):
    for _ in range(size):
        if tier == "Low":
            life_exp = np.random.normal(56, 4)
            mean_school = np.random.normal(3.5, 1.2)
            exp_school = np.random.normal(8.5, 1.5)
            gni = np.random.normal(2500, 800)
        elif tier == "Medium":
            life_exp = np.random.normal(66, 3)
            mean_school = np.random.normal(7.0, 1.3)
            exp_school = np.random.normal(11.5, 1.5)
            gni = np.random.normal(8000, 2000)
        elif tier == "High":
            life_exp = np.random.normal(74, 2.5)
            mean_school = np.random.normal(9.5, 1.2)
            exp_school = np.random.normal(13.5, 1.3)
            gni = np.random.normal(18000, 4000)
        else:  # Very High
            life_exp = np.random.normal(80, 2)
            mean_school = np.random.normal(12.5, 1.0)
            exp_school = np.random.normal(16.5, 1.2)
            gni = np.random.normal(42000, 12000)

        life_exp = np.clip(life_exp, 40, 90)
        mean_school = np.clip(mean_school, 0, 15)
        exp_school = np.clip(exp_school, 4, 20)
        gni = np.clip(gni, 500, 120000)

        rows.append([countries[idx], life_exp, mean_school, exp_school, gni])
        idx += 1

df = pd.DataFrame(
    rows,
    columns=[
        "Country",
        "Life_Expectancy",
        "Mean_Years_Schooling",
        "Expected_Years_Schooling",
        "GNI_Per_Capita",
    ],
)

# --- Build HDI using the (simplified) UNDP dimension-index formula ---
# Life expectancy index: min 20, max 85
li = (df["Life_Expectancy"] - 20) / (85 - 20)

# Education index: average of mean-years (0-15) and expected-years (0-18) sub-indices
mysi = df["Mean_Years_Schooling"] / 15
eysi = df["Expected_Years_Schooling"] / 18
ei = (mysi + eysi) / 2

# Income index: log-scaled, min $100, max $75,000
ii = (np.log(df["GNI_Per_Capita"]) - np.log(100)) / (np.log(75000) - np.log(100))

hdi = (li * ei * ii) ** (1 / 3)
hdi = np.clip(hdi, 0, 1)

# add a little noise so the model has something real to learn
hdi = hdi + np.random.normal(0, 0.01, size=len(hdi))
df["HDI_Score"] = np.clip(hdi, 0, 1).round(3)

# Introduce a few missing values to demonstrate the preprocessing step
for col in ["Life_Expectancy", "Mean_Years_Schooling", "GNI_Per_Capita"]:
    missing_idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[missing_idx, col] = np.nan

output_path = DATA_DIR / "hdi_dataset.csv"
df.to_csv(output_path, index=False)
print(f"Dataset saved -> {output_path}")
print(df.head())
print(df.isnull().sum())
