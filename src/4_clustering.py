import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv('spotify_cleaned.csv')

# Keep a copy of original dataframe
df_original = df.copy()

# Remove non-numeric columns like 'playlist_genre' before clustering
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Fill any remaining missing values if present (just in case)
numeric_df = numeric_df.fillna(numeric_df.mean())

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_df)

# Find optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('📌 Elbow Method to Determine Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Choose optimal clusters (e.g., 4 based on elbow graph)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster label to original dataframe
df_original['Cluster'] = clusters

# Pick two meaningful features for 2D cluster visualization
x_feature = numeric_df.columns[0]
y_feature = numeric_df.columns[1]

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df_original[x_feature],
    y=df_original[y_feature],
    hue=df_original['Cluster'],
    palette='Set2',
    s=80
)
plt.title(f'🎯 Clustering of Songs Using "{x_feature}" and "{y_feature}"')
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.legend(title='Cluster')
plt.grid()
plt.show()
