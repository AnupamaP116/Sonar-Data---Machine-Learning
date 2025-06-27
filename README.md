# Sonar-Data---Machine-Learning
# 🌊 Sonar Signal Classification using Machine Learning

## 📌 Project Overview

This project uses machine learning to classify sonar signals bounced off metal cylinders or rocks on the ocean floor. The goal is to build a robust classification model that can accurately distinguish between **"Mines"** and **"Rocks"** using the frequency data captured by 60 sonar sensors.

---

## 📁 Files in the Repository

- `Sonar Data.ipynb` – Jupyter Notebook containing full EDA, preprocessing, and classification model building
- `README.md` – Documentation of the project

---

## 📊 Dataset Description

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))
- **Attributes**: 60 numerical features representing sonar signal strength at various frequencies
- **Target**:  
  - `M`: Mine  
  - `R`: Rock

---

## 🧠 Tech Stack Used

- **Python**
- **Jupyter Notebook**
- **Libraries**:
  - `Pandas`, `NumPy` – Data manipulation
  - `Matplotlib`, `Seaborn` – Data visualization
  - `Scikit-learn` – Machine learning models, preprocessing, evaluation

---

## 🔍 Workflow Summary

1. **Data Loading**  
   Loaded sonar frequency data into a Pandas DataFrame.

2. **Exploratory Data Analysis (EDA)**  
   - Understood data shape, value ranges, and class distribution
   - Visualized correlations using heatmaps

3. **Preprocessing**  
   - Encoded target variable (`M` = 1, `R` = 0)
   - Standardized input features for better model performance

4. **Model Building**  
   Trained and evaluated the following ML models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Decision Tree
   - Random Forest
   - Gradient Boosting

5. **Model Evaluation**  
   - Accuracy Score  
   - Confusion Matrix  
   - Classification Report (Precision, Recall, F1-Score)

---

## ✅ Results & Observations

- **Best Performing Model**: *e.g., Random Forest (replace with actual result)*
- **Accuracy Achieved**: *e.g., 90.38% (replace with actual accuracy from your notebook)*
- **Insight**: Sonar signals are effectively separable using frequency data. Tree-based models like Random Forest performed best due to their ability to handle complex patterns.

---

## 🚀 How to Run

### 🔧 Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
