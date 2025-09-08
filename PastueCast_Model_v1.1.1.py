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
from sklearn.compose import ColumnTransformer

df = pd.read_csv("PastureCastModel_V1.1.1.csv")
df.dtypes
df.isnull().sum()

# Ensure the 'Date' column is in datetime format (handles mixed formats)
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df['Julian Date'] = df['Date'].dt.dayofyear

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


# Define the independent variables (features) and the target variable
features = ['MeanHeight(mm)', 'Julian_Cos', 'SAVI_mean', 'EVI_mean', 'NDVI_mean', 'NDRE_mean', 'Avg_21D_SWB_frac']
target = 'Biomass(kg/ha)'

# Ensure your data is clean and handle missing values
X = df[features].round(3)
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Preprocessing pipeline - scaling
preprocessor = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# --- Ridge regression and hyperparameter tuning ---
param_grid = {'alpha': np.logspace(-2, 2, 10)}  # Exploring different alpha values
ridge = Ridge()

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_preprocessed, y_train)

# Always register your model into variable named "model"
model = grid_search.best_estimator_

# Predict on test set
y_pred = model.predict(X_test_preprocessed).round(0)

# Evaluation Metrics
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = test_mse ** 0.5 
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

# Print evaluation results
print(f"\nBest Alpha: {grid_search.best_params_['alpha']}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R² Score: {test_r2:.4f}")

# Define the path ----- Change Accodingly
save_path = r"Biomass Lab work\Testbed_2025TB_Model_V1.1.1\PastureCastModel_V1.1.1_Testing"

# Make sure the folder exists (creates it if it doesn't)
os.makedirs(save_path, exist_ok=True)

# File names
model_file = os.path.join(save_path, "ridge_model_aug.joblib")
scaler_file = os.path.join(save_path, "scaler_aug.joblib")

# Save model and scaler
joblib.dump(model, model_file)
joblib.dump(preprocessor, scaler_file)

print(f"Trained Ridge model saved to {model_file}")
print(f"Scaler saved to {scaler_file}")

shutil.copy(os.path.join(save_path, "ridge_model_aug.joblib"), "./ridge_model_aug.joblib")
shutil.copy(os.path.join(save_path, "scaler_aug.joblib"), "./scaler_aug.joblib")




