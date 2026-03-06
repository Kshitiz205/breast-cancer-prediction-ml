
# Breast Cancer Prediction API (FastAPI + Machine Learning)

## Overview

Live API link (deployed on render) : https://breast-cancer-prediction-ml-3.onrender.com/docs
This project is a simple **machine learning API that predicts whether a breast tumor is Benign or Malignant** using features from the Breast Cancer dataset.

The model is trained using **scikit-learn** and **XGBoost**, saved as a `.pkl` file, and served through a **FastAPI backend** so that predictions can be made through an API request.

The goal of this project is to understand how a **machine learning model can be deployed as a real API**, which is an important step in real-world ML applications.

----------

# Project Structure

```
breast-cancer-api
│
├── app
│   └── app.py
│
├── data
│   └── breast_cancer.csv
│
├── models
│   └── breast_cancer_model.pkl
│
├── notebooks
│   └── breast_cancer_prediction.ipynb
│
├── requirements.txt
└── README.md
```
**app/**  
Mostly contains the FastAPI code.

**data/**  
Contains the dataset csv file on which the model was trained.

**models/**  
Contains the trained machine learning model.

**notebooks/**  
Contains the ipynb notebook file having EDA analysis, model training, pipeline codes.

**requirements.txt**  
All required Python libraries.

----------

# Model Details

The model was trained on the **Breast Cancer Classification** dataset from kaggle.

### Target Classes

-   **1 → Malignant (Cancerous)**
    
-   **0 → Benign (Non Cancerous)**
    

The model predicts the class along with a **confidence score**.

Example API response:

```json
{
  "prediction": "Malignant",
  "confidence": 0.89
}
```
4 different models were trained and accuracies were compared:-
 **1. Logistic Regression**
 **2. Decision Tree**
 **3. Random Forest**
 **4. XGBoost**

Below are the model accuracy scores:

```json
Logistic Regression : 0.9736842105263158 
Decision Tree : 0.9385964912280702 
Random Forest : 0.9649122807017544 
XGBoost : 0.956140350877193
```
Therefore, Logistic Regression was chosen as the final model with the performance metrics on the training dataset below:-

```json
            precision  recall  f1-score  support 
0                0.97    0.99      0.98       71 
1                0.98    0.95      0.96       43 

accuracy                           0.97      114 
macro avg        0.97    0.97      0.97      114 
weighted avg     0.97    0.97      0.97      114

True Negatives (TN) = 70
False Positives (FP) = 1
False Negatives (FN) = 2
True Positives (TP) = 41
```

----------

# Feature Importance

After training the model, feature importance was analyzed to see which features influence predictions the most.

The **top important features (in descending order)** are:

1.  texture_worst
    
2.  radius_se
    
3.  symmetry_worst
    
4.  concave points_mean
    
5.  concavity_worst
    
6.  area_se
    
7.  radius_worst
    
8.  area_worst
    
9.  concavity_mean
    
10.  concave points_worst
    

These features contribute the most to the model's decision making.

A **feature importance bar chart** and **correlation heatmap** were also generated to visualize relationships between features.

----------

# Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

----------

### 2. Run the FastAPI server

```bash
uvicorn main:app --reload
```
or
```bash
python -m uvicorn app.app:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```
Go to this link:

```
http://127.0.0.1:8000/docs
```

----------

# API Endpoint

### Predict Tumor Type

**POST**

```
/predict

```

Example request body:

```json
{
"radius_mean": 14.1,
"texture_mean": 20.5,
"perimeter_mean": 92.0,
"area_mean": 654.0,
"smoothness_mean": 0.096,
"compactness_mean": 0.104,
"concavity_mean": 0.089,
"concave_points_mean": 0.048,
"symmetry_mean": 0.181,
"fractal_dimension_mean": 0.062,
"radius_se": 0.405,
"texture_se": 1.217,
"perimeter_se": 2.866,
"area_se": 40.337,
"smoothness_se": 0.007,
"compactness_se": 0.021,
"concavity_se": 0.030,
"concave_points_se": 0.011,
"symmetry_se": 0.018,
"fractal_dimension_se": 0.003,
"radius_worst": 16.5,
"texture_worst": 28.1,
"perimeter_worst": 108.0,
"area_worst": 870.0,
"smoothness_worst": 0.132,
"compactness_worst": 0.254,
"concavity_worst": 0.280,
"concave_points_worst": 0.114,
"symmetry_worst": 0.290,
"fractal_dimension_worst": 0.083
}

```

Response:

```json
```json
{
  "prediction": "Malignant",
  "confidence": 0.6923866381966085
}
```

----------

# Visualizations

Two visualizations were generated during the project:

**1. Correlation Heatmap**  
Shows how different features in the dataset are correlated with each other.

**2. Feature Importance Plot**  
Displays which features the trained model relies on the most for predictions.

These help in understanding the model's behavior and improving interpretability.

----------

# Why This Project Matters

This project demonstrates a **complete mini ML deployment pipeline**:

-   Data → Model Training
    
-   Model Saving (`.pkl`)
    
-   API Development with FastAPI
    
-   Real-time Prediction
    

It is a good starting point for learning how **machine learning models are turned into real services**.

----------

# Possible Improvements

Some ways this project can be extended:

-   Add **input validation with Pydantic**
    
-   Build a **simple frontend interface**
    
-   Deploy the API using **Docker**
    
-   Deploy to **AWS / Render / Railway**
    
-   Add **model monitoring and logging**
-   Improve model parameters using **grid search**
    

----------

# Author
Kshitiz Singh

(Built as a **learning project for Machine Learning model deployment using FastAPI**)
