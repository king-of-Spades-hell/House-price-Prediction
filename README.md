import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Load the data
data = pd.read_csv("housing.csv")

# Handle missing values
data.dropna(inplace=True)

# Define features and target
x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Join features and target for training data
train_data = X_train.join(Y_train)

# Convert categorical 'ocean_proximity' to dummy variables
train_data = train_data.join(pd.get_dummies(train_data['ocean_proximity'])).drop(['ocean_proximity'], axis=1)

# Feature engineering
train_data["bedroom_ratio"] = train_data["total_bedrooms"] / train_data["total_rooms"]
train_data["household_rooms"] = train_data["total_rooms"] / train_data["households"]

# Define the feature set (X) and target (y) for training
x_train = train_data.drop(['median_house_value'], axis=1)
y_train = train_data['median_house_value']

# Linear Regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Process the test data
test_data = X_test.join(Y_test)
test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)

# Convert categorical 'ocean_proximity' to dummy variables in test set
test_data = test_data.join(pd.get_dummies(test_data['ocean_proximity'])).drop(['ocean_proximity'], axis=1)

# Feature engineering for test data
test_data["bedroom_ratio"] = test_data["total_bedrooms"] / test_data["total_rooms"]
test_data["household_rooms"] = test_data["total_rooms"] / test_data["households"]

# Ensure that test data has the same features as training data
x_test = test_data[x_train.columns]
y_test = test_data['median_house_value']

# Evaluate the Linear Regression model
print("Linear Regression score:", reg.score(x_test, y_test))

# Random Forest model
forest = RandomForestRegressor()
forest.fit(x_train, y_train)

# Evaluate the Random Forest model
print("Random Forest score:", forest.score(x_test, y_test))

# Perform grid search for Random Forest
param_grid = {
    "n_estimators": [3, 10, 30],  # Corrected spelling
    "max_features": [2, 4, 6, 8]
}

grid_search = GridSearchCV(forest, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(x_train, y_train)

# Best estimator from grid search
best_forest = grid_search.best_estimator_

# Evaluate the best Random Forest model
print("Best Random Forest score:", best_forest.score(x_test, y_test))
