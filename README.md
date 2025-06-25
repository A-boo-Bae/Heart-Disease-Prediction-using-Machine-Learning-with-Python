# 💓 Heart Disease Prediction using Machine Learning

This repository contains a **machine learning project** that predicts the likelihood of heart disease using various health indicators. The project employs multiple classification algorithms and provides a comparative analysis of their performances.

---

## 📑 Table of Contents
- [🧠 Project Overview](#project-overview)
- [📊 Dataset](#dataset)
- [🚀 Steps Taken](#steps-taken)
  - [1️⃣ Importing Datasets and Libraries](#1-importing-datasets-and-libraries)
  - [2️⃣ Data Collection and Processing](#2-data-collection-and-processing)
  - [3️⃣ Data Exploration](#3-data-exploration)
  - [4️⃣ Data Preprocessing](#4-data-preprocessing)
  - [5️⃣ Model Training and Evaluation](#5-model-training-and-evaluation)
  - [6️⃣ Comparing Models](#6-comparing-models)
  - [7️⃣ Building a Predictive System](#7-building-a-predictive-system)
- [🛠️ Requirements](#requirements)
- [💻 How to Run](#how-to-run)
- [📈 Results](#results)
- [🏷️ Hashtags](#hashtags)

---

## 🧠 Project Overview
The goal is to create a machine learning system capable of predicting the presence of heart disease based on patient health parameters. Multiple algorithms are trained, evaluated, and compared to identify the most effective approach.

---

## 📊 Dataset
The dataset used (`heart.csv`) includes the following features:

- `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, and `target`.

The `target` column indicates heart disease diagnosis:
- `0` = No disease
- `1` = Disease present

---

## 🚀 Steps Taken

### 1️⃣ Importing Datasets and Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Other imports: SVM, KNN, Decision Tree, Random Forest, Naive Bayes
```

---

### 2️⃣ Data Collection and Processing
```python
heart_data = pd.read_csv("path/to/heart.csv")
print(heart_data.head())
print(heart_data.tail())
print(heart_data.isnull().sum())
```
✅ No missing values detected.

---

### 3️⃣ Data Exploration
```python
heart_data.describe()
```
Basic statistical insights are generated.

---

### 4️⃣ Data Preprocessing
```python
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)
```

---

### 5️⃣ Model Training and Evaluation

#### Logistic Regression
```python
lr = LogisticRegression()
lr.fit(X_train, Y_train)
```
- Training Accuracy: **85.12%**

#### Support Vector Machine
```python
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, Y_train)
```
- Training Accuracy: **70.25%**

#### K-Nearest Neighbors
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
```
- Training Accuracy: **79.34%**

#### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
```
- Training Accuracy: **100.0%**

#### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train)
```
- Training Accuracy: **100.0%**

#### Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)
```
- Training Accuracy: **84.71%**

---

### 6️⃣ Comparing Models

| Model               | Test Accuracy (%) |
|--------------------|-------------------|
| Logistic Regression| 80.33             |
| Random Forest      | 80.33             |
| Naive Bayes        | 80.33             |
| Decision Tree      | 75.41             |
| SVM                | 62.30             |
| KNN                | 62.30             |

🏆 **Best performers**: Logistic Regression, Random Forest, and Naive Bayes.

---

### 7️⃣ Building a Predictive System
Example prediction using Logistic Regression:
```python
input_data = (56, 1, 1, 128, 236, 0, 1, 178, 0, 0.8, 2, 0, 2)
input_array = np.asarray(input_data).reshape(1, -1)

prediction = lr.predict(input_array)

if prediction[0] == 0:
    print("The person does NOT have heart disease.")
else:
    print("The person HAS heart disease.")
```

---

## 🛠️ Requirements
Install required libraries using:
```bash
pip install numpy pandas scikit-learn
```

---

## 💻 How to Run

1. Clone this repository:
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
```

2. Navigate to the directory:
```bash
cd heart-disease-prediction
```

3. Place `heart.csv` in the root directory.

4. Run the script:
```bash
python main.py
```

---

## 📈 Results

| Model               | Training Accuracy | Test Accuracy |
|--------------------|-------------------|---------------|
| Logistic Regression| 85.12%            | 80.33%        |
| SVM                | 70.25%            | 62.30%        |
| KNN                | 79.34%            | 62.30%        |
| Decision Tree      | 100.0%            | 75.41%        |
| Random Forest      | 100.0%            | 80.33%        |
| Naive Bayes        | 84.71%            | 80.33%        |

📌 **Logistic Regression** was chosen for deployment due to its balance of training and test performance.

---

## 🏷️ Hashtags

```
#MachineLearning #HeartDiseasePrediction #Python #DataScience #LogisticRegression 
#RandomForest #NaiveBayes #HealthcareAI #scikitLearn #HeartHealth #AIinHealthcare 
#ClassificationModels #MLProject #OpenSource #PredictiveAnalytics
