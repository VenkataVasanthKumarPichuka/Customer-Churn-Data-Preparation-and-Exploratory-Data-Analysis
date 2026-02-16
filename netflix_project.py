# ================================
# TASK 4: NETFLIX DATA ANALYSIS
# Cleaning + Feature Engineering + EDA
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading Netflix Dataset...")

# -----------------------------
# 1. DATA UNDERSTANDING & CLEANING
# -----------------------------
df = pd.read_csv("netflix_titles.csv")

print(df.info())

# Convert date_added to datetime
df["date_added"] = pd.to_datetime(
    df["date_added"].str.strip(),
    errors="coerce"
)

# Fill missing values
df["director"].fillna("Unknown", inplace=True)
df["cast"].fillna("Unknown", inplace=True)
df["country"].fillna("Unknown", inplace=True)

# -----------------------------
# 2. FEATURE ENGINEERING
# -----------------------------

df["release_year"] = df["release_year"]
df["is_movie"] = df["type"].apply(lambda x: 1 if x=="Movie" else 0)

df["release_decade"] = (df["release_year"] // 10) * 10

# Top 5 countries
top_countries = df["country"].value_counts().head(5)
print("\nTop Countries:")
print(top_countries)

# -----------------------------
# 3. VISUALIZATION & INSIGHTS
# -----------------------------

sns.set(style="whitegrid")

# Pie chart Movie vs TV Show
plt.figure()
df["type"].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Movies vs TV Shows")
plt.ylabel("")
plt.show()

# Line plot content per year
plt.figure()
df["release_year"].value_counts().sort_index().plot()
plt.title("Content Released per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

# Top Genres
plt.figure(figsize=(10,5))
df["listed_in"].str.split(", ").explode().value_counts().head(10).plot.bar()
plt.title("Top 10 Genres")
plt.show()

# Country vs Content Heatmap
country_data = df["country"].value_counts().head(10)
plt.figure()
sns.heatmap(country_data.to_frame(), annot=True, cmap="coolwarm")
plt.title("Top Countries Content Volume")
plt.show()

# -----------------------------
# 4. SAVE CLEANED DATA
# -----------------------------
df.to_csv("Cleaned_Netflix_Titles.csv", index=False)

print("\n✅ TASK 4 COMPLETED")
