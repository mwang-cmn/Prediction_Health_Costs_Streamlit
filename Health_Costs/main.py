import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Load and clean data
data = pd.read_csv('insurance.csv')
data.drop_duplicates(inplace=True)

# Copy data
health = data.copy()

# Label encoding
label = LabelEncoder()
health['sex'] = label.fit_transform(health['sex'])
health['smoker'] = label.fit_transform(health['smoker'])
health = pd.get_dummies(health, columns=['region'], drop_first=True)
health['bmi_category'] = pd.cut(health['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Healthy', 'Overweight', 'Obese'])
health['age_group'] = pd.cut(health['age'], bins=[0, 25, 40, 64, 100], labels=['Young Adult', 'Adult', 'Middle Aged', 'Senior'])
health = health.drop(['age', 'bmi'], axis=1)
bmi_order = ['Underweight', 'Healthy', 'Overweight', 'Obese']
age_order = ['Young Adult', 'Adult', 'Middle Aged', 'Senior']
health['bmi_category'] = pd.Categorical(health['bmi_category'], categories=bmi_order, ordered=True)
health['bmi_category_encoded'] = health['bmi_category'].cat.codes.replace(-1, np.nan) 

health['age_group'] = pd.Categorical(health['age_group'], categories=age_order, ordered=True)
health['age_group_encoded'] = health['age_group'].cat.codes.replace(-1, np.nan)

health.drop(['bmi_category', 'age_group'], axis=1, inplace=True)

# Define features and target variable
X = health.drop('expenses', axis=1)
y = health['expenses']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initial model training
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': list(range(2, 11)),  # Converted range to list
    'min_samples_split': list(range(2, 11))
}

rf_model_tuned = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_iter=20,  # Limit iterations
    random_state=42
)
rf_model_tuned.fit(X_train, y_train)

# Train final model with best parameters
params_rf = rf_model_tuned.best_params_
rf = RandomForestRegressor(
    n_estimators=params_rf['n_estimators'],
    max_depth=params_rf['max_depth'],
    min_samples_split=params_rf['min_samples_split'],
    min_samples_leaf=params_rf['min_samples_leaf'],
    max_features=params_rf['max_features']
)
rf.fit(X_train, y_train)

# Model evaluation
y_pred_rf2 = rf.predict(X_test)
r2_best_rf2 = r2_score(y_test, y_pred_rf2)
mae_best_rf2 = mean_absolute_error(y_test, y_pred_rf2)

# Save model
joblib.dump(rf, 'tuned_random_forest_model.pkl')
