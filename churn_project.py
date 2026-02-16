import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# LOAD DATA
df = pd.read_csv("Telco-Customer-Churn.csv")

print(df.info())
print(df.describe())

# DATA CLEANING
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = df["TotalCharges"].astype(float)

df = df.dropna()

# FEATURE ENGINEERING
def tenure_group(x):
    if x <= 12:
        return "0-12"
    elif x <= 24:
        return "13-24"
    elif x <= 48:
        return "25-48"
    else:
        return "48+"

df["TenureGroup"] = df["tenure"].apply(tenure_group)

df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"]

# Convert Yes/No to 1/0
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].replace({"Yes":1,"No":0})

# ONE HOT ENCODING
df = pd.get_dummies(df, drop_first=True)

# VISUALIZATION
sns.countplot(x="Churn", data=df)
plt.show()

sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.show()

# SAVE CLEANED DATA
df.to_csv("Cleaned_Telco_Churn.csv", index=False)

print("TASK 3 COMPLETED")
