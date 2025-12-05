import pandas as pd

# Load dataset
df = pd.read_csv('spotify dataset.csv')

# Display basic details
print("✅ Dataset Loaded")
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# Drop columns with too many missing values or unnecessary columns (if any)
# Example (customize if needed):
# df.drop(['track_id', 'artist_name'], axis=1, inplace=True)

# Fill missing numeric values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Save cleaned data
df.to_csv('spotify_cleaned.csv', index=False)
print("\n✅ Cleaned data saved as 'spotify_cleaned.csv'")
