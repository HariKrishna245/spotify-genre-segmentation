import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('spotify_cleaned.csv')

# Genre distribution
plt.figure(figsize=(10, 6))
df['playlist_genre'].value_counts().plot(kind='bar', color='coral')
plt.title('🎵 Number of Songs per Genre')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualizing pairplot of first few numerical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
sns.pairplot(df[numerical_cols[:4]])
plt.show()
