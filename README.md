# ✈️ Airline Passenger Satisfaction Analysis using Machine Learning

A machine learning project aimed at predicting passenger satisfaction based on flight experience, demographics, and service ratings. Built using Python, scikit-learn, and SMOTE to address class imbalance.

---

## 📌 Overview

Airline customer satisfaction is critical for improving service quality and retaining loyal travelers.  
This project builds a machine learning pipeline to classify whether a passenger is **“Satisfied”** or **“Neutral/Dissatisfied”** using features like:

- In-flight services (WiFi, Food, Cleanliness)
- Travel Class, Flight Distance
- Customer type, Age, Delay status

---

## 🎯 Objective

To develop a robust binary classification model that predicts airline passenger satisfaction using survey data.  
The model outputs:

- ✅ `Satisfied`
- ❌ `Neutral or Dissatisfied`

---

## 🗃 Dataset

- **Source:** Maven Analytics  
- **Format:** CSV  
- **Size:** 129,880 rows × 24 columns  
- **Access:** [Download via Google Drive](https://drive.google.com/file/d/1VtYC86HrBZNrX3-4E-wtQ6ntz469A0IF/view?usp=sharing)

### 🔎 Features Include:
- Demographics: Age, Gender  
- Travel Info: Travel Type, Class, Flight Distance, Delay Timings  
- Services Rated: Online Boarding, WiFi, Food, Seat Comfort  
- 🎯 **Target Column:** `Satisfaction`

---

## 🧰 Tools & Libraries

- **Language:** Python  
- **IDE:** Google Colab  
- **Data Handling:** `pandas`, `numpy`  
- **Visualization:** `matplotlib`, `seaborn`  
- **Modeling:** `scikit-learn`  
- **Class Imbalance:** `imbalanced-learn` (SMOTE)  
- **Model Persistence:** `joblib`

---

## 🔍 Workflow Summary

### ✅ Data Preprocessing
- Null handling with median imputation
- Outlier removal using IQR
- Feature encoding (OneHotEncoder)
- Feature scaling (StandardScaler)

### ✅ Exploratory Data Analysis
- Satisfaction trends by class, WiFi quality, age group
- Correlation heatmaps & target distribution plots

### ✅ Feature Selection
- Top 10 features selected using `SelectKBest` (mutual information)

### ✅ Class Imbalance Handling
- Applied **SMOTE** to balance the dataset (approx. 56% Satisfied, 44% Dissatisfied)

### ✅ Model Building
- Trained 5 algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
  - Gradient Boosting

### ✅ Hyperparameter Tuning
- Used `GridSearchCV` for all models  
- Evaluated using: Accuracy, Precision, Recall, F1, ROC-AUC

### ✅ Final Model: **Random Forest**
- Saved using `joblib` for future use

---

## 🏆 Model Performance (Random Forest)

| Metric        | Value      |
|---------------|------------|
| Accuracy      | 93.25%     |
| Precision     | 94.63%     |
| Recall        | 91.63%     |
| F1 Score      | 0.931      |
| ROC-AUC Score | 0.984      |

### 🔑 Top Features:
- Online Boarding
- Seat Comfort
- In-flight WiFi
- Type of Travel
- Class of Travel

---

## 📊 Visual Insights

Key charts and insights are included in the Jupyter notebook.  
Visuals include:

- Target distribution bar chart
- Correlation heatmap
- Satisfaction vs Travel Class
- ROC Curve
- Confusion Matrix
- Model comparison across metrics

---

## 💾 Repository Contents

- 📓 [View the Jupyter Notebook with code, visualizations, and analysis](PROJECT_AIRLINE_PASSENGER_SATISFACTION_ANALYSIS_USING_MACHINE_LEARNING.ipynb)

- `README.md` — Project overview, workflow, insights, and setup.
- Visuals embedded directly in the notebook.
- Dataset not included due to size — [Access it here](https://drive.google.com/file/d/1VtYC86HrBZNrX3-4E-wtQ6ntz469A0IF/view?usp=sharing).


