import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = "London dataset.csv"
df = pd.read_csv(file_path)

# Display basic info and first few rows
print(df.info())
print(df.head())

# Set style
sns.set(style="whitegrid")

# === Chart 1: Number of Restaurants per Borough ===
plt.figure(figsize=(10, 6))  # Set figure size for the first plot
borough_counts = df["borough"].value_counts().reset_index()
borough_counts.columns = ["borough", "count"]

sns.barplot(data=borough_counts, x="count", y="borough", hue="borough", palette="viridis", legend=False)
plt.title("Number of Restaurants per Borough")
plt.xlabel("Count")
plt.ylabel("Borough")

plt.tight_layout()
plt.savefig("Number of Restaurants per Borough.png")
plt.show()

# === Chart 2: Top 10 Cuisines in London ===
plt.figure(figsize=(10, 6))  # Set figure size for the second plot
top_cuisines = df["cuisine"].value_counts().head(10).reset_index()
top_cuisines.columns = ["cuisine", "count"]

sns.barplot(data=top_cuisines, x="count", y="cuisine", hue="cuisine", palette="magma", legend=False)
plt.title("Top 10 Cuisines in London Restaurants")
plt.xlabel("Count")
plt.ylabel("Cuisine")

plt.tight_layout()
plt.savefig("Top 10 Cuisines.png")
plt.show()

# 2. Halal vs Non-Halal Restaurant Count

# Combine 'Halal' and 'halal' columns and standardize values
df['is_halal'] = df[['Halal', 'halal']].bfill(axis=1).iloc[:, 0]
halal_counts = df['is_halal'].value_counts()

# 2. Halal vs Non-Halal Restaurant Count

# Combine 'Halal' and 'halal' columns and standardize values
df['is_halal'] = df[['Halal', 'halal']].bfill(axis=1).iloc[:, 0]
halal_counts = df['is_halal'].value_counts()

# Plot
plt.figure()
sns.barplot(x=halal_counts.index, y=halal_counts.values, palette="Set2")
plt.title("Halal vs Non-Halal Restaurant Count")
plt.xlabel("Halal Certified")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig("Halal vs Non-Halal Restaurant Count.png")
plt.show()

# Filter valid latitude and longitude entries
location_df = df[['lat', 'lng']].dropna()

# Scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=location_df, x='lng', y='lat', alpha=0.5, s=40, color='darkred')
plt.title("Restaurant Locations in London")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.tight_layout()
plt.savefig("Restaurant Locations in London.png")
plt.show()






