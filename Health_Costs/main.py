
# Import libraries
import numpy as np
import pandas as pd


# Import data
url = " https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv"
dataset = pd.read_csv(url)
dataset.tail()
data = dataset.copy()
data.drop_duplicates(inplace=True)
health = data.copy()
# label encode categorical columns

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
# Male: 1, Female: 0 - Sex
health['sex'] = label.fit_transform(health['sex'])
# Yes: 1, No: 0 - Smoker
health['smoker'] = label.fit_transform(health['smoker'])
health = pd.get_dummies(health, columns=['region'], drop_first=True)


# Ordered categories
bmi_order = ['Underweight', 'Healthy', 'Overweight', 'Obese']
age_order = ['Young Adult', 'Adult', 'Middle Aged', 'Senior']
# Convert categorical columns to ordered type
health['bmi_category'] = pd.Categorical(health['bmi_category'], categories=bmi_order, ordered=True)
health['age_group'] = pd.Categorical(health['age_group'], categories=age_order, ordered=True)
# Encode as numerical values
health['bmi_category_encoded'] = health['bmi_category'].cat.codes
health['age_group_encoded'] = health['age_group'].cat.codes
health.drop(['bmi_category', 'age_group'], axis=1, inplace=True)
X = health.drop('expenses', axis=1)
y = health['expenses']

# Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5,10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': range(2, 11),
    'min_samples_split': range(2, 11)
}

rf_model_tuned = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, cv=5, scoring='neg_mean_absolute_error')
rf_model_tuned.fit(X_train, y_train)
params_rf = rf_model_tuned.best_params_
rf = RandomForestRegressor(n_estimators=params_rf['n_estimators'],
                           max_depth=params_rf['max_depth'],
                           min_samples_split=params_rf['min_samples_split'],
                           min_samples_leaf = params_rf['min_samples_leaf'],
                           max_features = params_rf['max_features'])
rf.fit(X_train, y_train)
y_pred_rf2 = rf.predict(X_test)
r2_best_rf2 = r2_score(y_test, y_pred_rf2)
mae_best_rf2 = mean_absolute_error(y_test, y_pred_rf2)

import joblib
joblib.dump(rf, 'tuned_random_forest_model.pkl')

