import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('spotify_cleaned.csv')

# Encode genre as numbers
df['playlist_genre'] = df['playlist_genre'].astype('category').cat.codes

# Features and target
X = df.drop(['playlist_genre'], axis=1).select_dtypes(include=['float64', 'int64'])
y = df['playlist_genre']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Report
print("✅ Model Accuracy:", accuracy_score(y_test, predictions))
print("\n📊 Classification Report:\n", classification_report(y_test, predictions))
