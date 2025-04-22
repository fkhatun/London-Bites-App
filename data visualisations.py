import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = "London dataset.csv"
df = pd.read_csv(file_path)

# Display basic info and first few rows
print(df.info())
print(df.head())

# Set seaborn style
sns.set(style="whitegrid")

# === Chart 1: Number of Restaurants per Borough ===
plt.figure(figsize=(10, 6))
borough_counts = df["borough"].value_counts().reset_index()
borough_counts.columns = ["borough", "count"]

sns.barplot(data=borough_counts, x="count", y="borough", hue="borough", palette="viridis", legend=False)
plt.title("Number of Restaurants per Borough")
plt.xlabel("Count")
plt.ylabel("Borough")

plt.tight_layout()
plt.savefig("Number_of_Restaurants_per_Borough.png")
plt.show()

# === Chart 2: Top 10 Cuisines in London ===
plt.figure(figsize=(10, 6))
top_cuisines = df["cuisine"].value_counts().head(10).reset_index()
top_cuisines.columns = ["cuisine", "count"]

sns.barplot(data=top_cuisines, x="count", y="cuisine", hue="cuisine", palette="magma", legend=False)
plt.title("Top 10 Cuisines in London Restaurants")
plt.xlabel("Count")
plt.ylabel("Cuisine")

plt.tight_layout()
plt.savefig("Top_10_Cuisines.png")
plt.show()

# === Chart 3: Halal vs Non-Halal Restaurant Count ===

# Combine 'Halal' and 'halal' columns (handle casing inconsistency)
if 'Halal' in df.columns and 'halal' in df.columns:
    df['is_halal'] = df[['Halal', 'halal']].bfill(axis=1).iloc[:, 0]
elif 'halal' in df.columns:
    df['is_halal'] = df['halal']
elif 'Halal' in df.columns:
    df['is_halal'] = df['Halal']
else:
    raise KeyError("No 'halal' or 'Halal' column found in the dataset.")

# Standardize values (to lowercase and consistent tags)
df['is_halal'] = df['is_halal'].astype(str).str.strip().str.lower()
df['is_halal'] = df['is_halal'].replace({
    'yes': 'halal', 'true': 'halal', '1': 'halal',
    'no': 'non-halal', 'false': 'non-halal', '0': 'non-halal'
})
df['is_halal'] = df['is_halal'].where(df['is_halal'].isin(['halal', 'non-halal']), 'unknown')

# Count values
halal_counts = df['is_halal'].value_counts()

# Plot
plt.figure(figsize=(6, 4))
sns.barplot(x=halal_counts.index, y=halal_counts.values, palette="Set2")
plt.title("Halal vs Non-Halal Restaurant Count")
plt.xlabel("Halal Certified")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig("Halal_vs_Non-Halal_Restaurant_Count.png")
plt.show()

# === Chart 4: Restaurant Locations Scatter Plot ===

# Check if lat/lng columns exist
if 'lat' in df.columns and 'lng' in df.columns:
    location_df = df[['lat', 'lng']].dropna()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=location_df, x='lng', y='lat', alpha=0.5, s=40, color='darkred')
    plt.title("Restaurant Locations in London")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Restaurant_Locations_in_London.png")
    plt.show()
else:
    print("Latitude and longitude columns not found in dataset.")
