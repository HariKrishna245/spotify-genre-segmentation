# 🎵 Spotify Songs’ Genre Segmentation & Prediction  
*A Machine Learning + Streamlit Project*

---

## 📌 Overview  
This project performs **genre segmentation** and **genre prediction** for Spotify songs using machine learning.  
It includes:

- 🧹 Data Preprocessing  
- 📊 Exploratory Data Analysis (EDA)  
- 🔥 Correlation Heatmap  
- 🎛️ K-Means Clustering (Segmentation)  
- 🤖 Random Forest Genre Classification  
- 🌐 Streamlit Web App  

---

## 🚀 Features

### **1. Data Preprocessing**
- Handle missing values  
- Clean audio feature columns  
- Save cleaned dataset for modeling  

### **2. Exploratory Data Analysis**
- Bar chart of song distribution across genres  
- Pairplots for exploring patterns  

### **3. Correlation Analysis**
- Heatmap showing relationships between audio features  
- Identify strong & weak correlations  

### **4. Clustering (Segmentation)**
- K-Means clustering  
- Visual cluster separation  
- Helps understand grouping patterns of songs  

### **5. Genre Classification**
- RandomForestClassifier  
- ~56% accuracy  
- Precision/Recall/F1 for each genre label  

### **6. Streamlit Web App**
- Sliders to input audio features  
- Predicts the most likely genre  
- Displays training dataset  
- Shows model information  

---

## 📁 Project Structure  

```
spotify-genre-segmentation/
│-- 1_data_preprocessing.py
│-- 2_data_visualization.py
│-- 3_correlation_analysis.py
│-- 4_clustering.py
│-- 5_model_building.py
│-- app.py
│-- spotify dataset.csv
│-- spotify_cleaned.csv
│-- requirements.txt
│-- README.md
│-- .gitignore
```

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/HariKrishna245/spotify-genre-segmentation.git
cd spotify-genre-segmentation
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🎮 Run the Streamlit App

```bash
streamlit run app.py
```

Open in your browser:

```
http://localhost:8501/
```

---

## 🔥 Future Improvements  
- Hyperparameter tuning  
- Neural network classifier  
- PCA visualization  
- Deployment to Streamlit Cloud  

---



