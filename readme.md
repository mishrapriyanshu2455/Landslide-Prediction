# 🌍 Landslide Risk Prediction System

A machine learning web application that predicts landslide risk based on environmental and geological parameters. Built with a Random Forest classifier and deployed via an interactive Streamlit dashboard.

---

## 📌 Overview

Landslides pose serious threats to lives and infrastructure in hilly and mountainous regions. This project uses a synthetically generated dataset modeled on real-world risk factors to train a classification model that estimates the probability of a landslide occurring given a set of environmental conditions.

The system is optimized for **high recall** — minimizing false negatives is critical in a risk prediction context.

---

## 🗂️ Project Structure

```
├── dataset.py                     # Generates the synthetic training dataset
├── model.py                       # Trains, tunes, and evaluates the ML model
├── app.py                         # Streamlit web application
├── synthetic_landslide_dataset.csv  # Generated dataset (6000 samples)
├── landslide_model.pkl            # Saved trained model pipeline
├── features.pkl                   # Saved feature column list
├── Librairies.txt                 # Required Python libraries
└── README.md
```

---

## ⚙️ How It Works

### 1. Dataset Generation (`dataset.py`)
- Generates **6,000 synthetic samples** across three terrain types: Plains, Hills, and Mountains.
- Each terrain type has realistic distributions for rainfall, slope, elevation, vegetation, earthquake intensity, and soil type.
- A **risk score** is computed as a weighted combination of these features, with added noise to simulate real-world uncertainty.
- Samples with a risk score above a threshold are labeled as landslide events (~33% positive class).

**Risk Score Formula:**
```
risk_score = 0.20×(rainfall) + 0.20×(slope) + 0.15×(soil_risk)
           + 0.10×(elevation) + 0.15×(earthquake) + 0.20×(moisture) − 0.20×(vegetation)
```

### 2. Model Training (`model.py`)
- Features are one-hot encoded for terrain and soil type.
- A **Random Forest Classifier** is trained inside a `sklearn` Pipeline.
- **GridSearchCV** (5-fold cross-validation, scored on recall) is used to find optimal hyperparameters.
- A custom **probability threshold of 0.35** (instead of default 0.5) is applied to further boost recall.

### 3. Web App (`app.py`)
- Interactive sidebar lets users input environmental conditions.
- Displays prediction result, probability score, risk interpretation, and a progress bar.
- Shows a **feature importance chart** from the trained model.

---

## 🚀 Getting Started

### Prerequisites

Install the required libraries:

```bash
pip install numpy pandas streamlit joblib matplotlib scikit-learn
```

### Step 1 — Generate the Dataset

```bash
python dataset.py
```

This creates `synthetic_landslide_dataset.csv`.

### Step 2 — Train the Model

```bash
python model.py
```

This trains the model, prints evaluation metrics, and saves `landslide_model.pkl` and `features.pkl`.

### Step 3 — Launch the App

```bash
streamlit run app.py
```

Open the URL shown in your terminal (typically `http://localhost:8501`).

---

## 🧪 Model Performance

The model is evaluated using:
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)
- **Accuracy Score**

> Optimized for **recall** to minimize missed landslide predictions (false negatives).

---

## 📊 Input Features

| Feature | Description | Range |
|---|---|---|
| Rainfall | Precipitation in mm | 0 – 300 |
| Slope | Terrain slope in degrees | 0 – 50 |
| Elevation | Height above sea level in meters | 0 – 4000 |
| Vegetation Index | Normalized vegetation cover | 0 – 1 |
| Earthquake Intensity | Seismic activity level | 0 – 7 |
| Moisture Retention | Soil moisture level | 0 – 1 |
| Terrain Type | Plains / Hills / Mountains | Categorical |
| Soil Type | Clay / Sand / Rock | Categorical |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| ML Framework | scikit-learn |
| Web App | Streamlit |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Model Persistence | joblib |

---

## 📝 Notes

- The dataset is **synthetically generated** and intended for demonstration and educational purposes.
- The 0.35 classification threshold is deliberately conservative to favour safety-critical recall over precision.
- To retrain on real geospatial data, replace `dataset.py` with your own data pipeline and re-run `model.py`.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).