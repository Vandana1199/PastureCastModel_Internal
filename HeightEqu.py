#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import joblib
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


# In[2]:


df = pd.read_csv("PastureCastModel_V1.1.1.csv")
df


# In[3]:


df.dtypes


# In[4]:


df.isnull().sum()


# In[5]:


# Ensure the 'Date' column is in datetime format (handles mixed formats)
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df['Julian Date'] = df['Date'].dt.dayofyear
df


# In[6]:


# 1️⃣ Define the function
def julian_cos_encoding_from_date(df: pd.DataFrame, date_col: str, cos_JD: str = "Julian_Cos") -> pd.DataFrame:
    """
    Takes a real date column, extracts year and day-of-year,
    checks leap year, converts to radians, then applies cosine conversion.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    years = df[date_col].dt.year
    julian_days = df[date_col].dt.dayofyear
    days_in_year = np.where(
        ((years % 4 == 0) & (years % 100 != 0)) | (years % 400 == 0),
        366, 365
    )
    radians = 2 * np.pi * julian_days / days_in_year
    df[cos_JD] = np.cos(radians)
    return df

# 2️⃣ Apply the function BEFORE defining your features
df = julian_cos_encoding_from_date(df, date_col="Date")


# In[7]:


df.columns


# In[12]:


# Define the independent variables (features) and the target variable
features = ['MeanHeight(mm)']
target = 'Biomass(kg/ha)'

# Ensure your data is clean and handle missing values
X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# # Preprocessing pipeline - Include all your preprocessing steps in below function to be applied to new data
# preprocessor = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])

# # have variable names as shown below _preprocessed
# # Apply preprocessing
# X_train_preprocessed = preprocessor.fit_transform(X_train)
# X_test_preprocessed = preprocessor.transform(X_test)

# --- Ridge regression and hyperparameter tuning ---
param_grid = {'alpha': np.logspace(-2, 2, 10)}  # Exploring different alpha values
ridge = Ridge()

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Always register your model into variable named "model"
model = grid_search.best_estimator_

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation Metrics (using absolute predictions)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = test_mse ** 0.5 
test_mae = mean_absolute_error(y_test, y_pred)  # MAE
test_r2 = r2_score(y_test, y_pred)  # R² Score

# Print evaluation results
print(f"Best Alpha: {grid_search.best_params_['alpha']}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R² Score: {test_r2:.4f}")

# --- Print Ridge regression equation with feature names ---
feature_names = X_train.columns  # assumes X_train is a DataFrame
coefs = model.coef_
intercept = model.intercept_

eq_terms = [f"({coef:.3f} * {name})" for coef, name in zip(coefs, feature_names)]
equation = " + ".join(eq_terms)
equation = f"y = {intercept:.3f} + {equation}"

print("\nRidge Regression Equation with Feature Names:")
print(equation)


# In[ ]:





# In[13]:


# Define the path ----- Change Accodingly
save_path = r"Biomass Lab work\Testbed_2025TB_Model_V1.1.1\PastureCastModel_V1.1.1_Testing"

# Make sure the folder exists (creates it if it doesn't)
os.makedirs(save_path, exist_ok=True)

# File names
model_file = os.path.join(save_path, "ridge_model.joblib")
scaler_file = os.path.join(save_path, "scaler.joblib")

# Save model and scaler
joblib.dump(best_ridge, model_file)
joblib.dump(scaler, scaler_file)

print(f"Trained Ridge model saved to {model_file}")
print(f"Scaler saved to {scaler_file}")


# In[ ]:


shutil.copy(os.path.join(save_path, "ridge_model.joblib"), "./ridge_model.joblib")
shutil.copy(os.path.join(save_path, "scaler.joblib"), "./scaler.joblib")


# In[ ]:




