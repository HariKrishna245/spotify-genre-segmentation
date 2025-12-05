import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page settings
st.set_page_config(page_title="Spotify Genre Predictor", layout="wide")

st.title("🎵 Spotify Songs' Genre Segmentation & Prediction")

# Load dataset
df = pd.read_csv('spotify_cleaned.csv')

# Encode target
df['playlist_genre'] = df['playlist_genre'].astype('category')
le = LabelEncoder()
df['playlist_genre_code'] = le.fit_transform(df['playlist_genre'])

# Prepare features and labels
X = df.drop(['playlist_genre', 'playlist_genre_code'], axis=1).select_dtypes(include=['float64', 'int64'])
y = df['playlist_genre_code']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sidebar sliders for feature input
st.sidebar.header("🎛️ Input Audio Features")
input_data = {}

for feature in X.columns:
    input_data[feature] = st.sidebar.slider(
        f"{feature}",
        float(df[feature].min()),
        float(df[feature].max()),
        float(df[feature].mean())
    )

# Predict genre
if st.sidebar.button("🎧 Predict Genre"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    genre_label = le.inverse_transform([prediction])[0]
    st.success(f"✅ **Predicted Genre: {genre_label}**")

# Display dataset
with st.expander("📊 View Training Dataset"):
    st.dataframe(df.head(20))

# Show model info
with st.expander("ℹ️ Model Info"):
    st.write("Model used: Random Forest Classifier")
    st.write(f"Training Data Shape: {X.shape}")
    st.write("Target: `playlist_genre`")
