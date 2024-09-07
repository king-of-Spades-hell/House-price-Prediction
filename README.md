Housing Price Prediction
This project uses machine learning techniques to predict housing prices based on various features such as geographical location, housing characteristics, and proximity to the ocean. The models used in this project include Linear Regression and Random Forest Regressor, with hyperparameter tuning performed on the Random Forest model using Grid Search.

Dataset
The dataset used for this project includes features like:

longitude: Longitude coordinate of the block.
latitude: Latitude coordinate of the block.
housing_median_age: Median age of the houses in the block.
total_rooms: Total number of rooms in the block.
total_bedrooms: Total number of bedrooms in the block.
population: Total population in the block.
households: Total number of households in the block.
median_income: Median income of households in the block (in tens of thousands of dollars).
median_house_value: Median house value in the block (in US dollars).
ocean_proximity: Categorical feature representing the proximity to the ocean.
Project Workflow
Data Preprocessing

Missing values are removed from the dataset using dropna().
Categorical variables (ocean_proximity) are converted into dummy variables using one-hot encoding.
New features are created:
bedroom_ratio: The ratio of total bedrooms to total rooms.
household_rooms: The average number of rooms per household.
Splitting the Data

The data is split into training (80%) and test sets (20%) using train_test_split().
Linear Regression Model

A linear regression model is trained on the training data and evaluated on the test data.
Random Forest Model

A random forest regressor is trained on the training data and evaluated on the test data.
Grid Search is used to find the best hyperparameters for the random forest model, optimizing for the lowest mean squared error.
Feature Transformation for Test Data

Log transformations are applied to some features in the test data to normalize distributions and handle outliers.
The test data is transformed similarly to the training data, ensuring feature consistency.
Requirements
Python 3.x
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
You can install the required Python packages using the following command:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
Usage
Prepare the Dataset: Ensure the dataset file housing.csv is present in the same directory as the script.
Run the Script: Execute the Python script to train models and evaluate their performance.
bash
Copy code
python housing_price_prediction.py
Model Evaluation: The script will output the following:
Linear Regression score on the test set.
Random Forest score on the test set.
Best Random Forest score using Grid Search.
Hyperparameter Tuning
Grid Search is applied to the Random Forest model to find the best set of hyperparameters:

n_estimators: Number of trees in the forest (3, 10, 30).
max_features: Number of features considered for splitting at each node (2, 4, 6, 8).
