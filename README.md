# Housing Price Prediction

This project implements a pipeline to predict housing prices using Linear Regression and Random Forest models. The dataset used is the `housing.csv` file, and it includes feature engineering, model training, evaluation, and hyperparameter optimization.

## 🚀 Features
- **Data Preprocessing**: Handles missing values and categorical data.
- **Feature Engineering**: Adds derived features to improve model performance.
- **Model Training**: Trains both Linear Regression and Random Forest models.
- **Hyperparameter Tuning**: Optimizes Random Forest parameters using GridSearchCV.
- **Evaluation**: Provides performance metrics for both models.

## 📂 Dataset
The dataset should be in CSV format and include the following columns:
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `ocean_proximity` (categorical)
- `median_house_value` (target)

## 🛠 Setup

### Prerequisites
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository.git
   cd your-repository
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the `housing.csv` dataset in the project directory.

## ▶️ Usage
Run the script to preprocess the data, train models, and evaluate performance:
```bash
python housing_price_prediction.py
```

## 📚 Key Steps

### 1. Data Preprocessing
- **Handle Missing Values**: Removes rows with missing data.
- **Categorical Variables**: Converts `ocean_proximity` into dummy variables.
- **Log Transformation**: Applies log transformation to numerical features for better normalization.

### 2. Feature Engineering
- Adds derived features:
  - `bedroom_ratio`: Ratio of `total_bedrooms` to `total_rooms`.
  - `household_rooms`: Average number of rooms per household.

### 3. Model Training and Evaluation
- **Linear Regression**:
  - Fits a linear model to the training data.
  - Evaluates performance using R² score.
- **Random Forest**:
  - Trains a Random Forest model for regression.
  - Evaluates performance using R² score.

### 4. Hyperparameter Tuning
- Performs GridSearchCV for Random Forest to find optimal values for:
  - `n_estimators`
  - `max_features`
- Evaluates the best model on the test set.

## 🔧 Customization
- **Add more features**: Include additional derived features to improve model accuracy.
- **Hyperparameter tuning**: Expand the parameter grid for GridSearchCV to explore more options.
- **Dataset**: Replace with a different dataset to explore other regression problems.

## 📝 Results
- **Linear Regression Score**: R² score for the Linear Regression model.
- **Random Forest Score**: R² score for the Random Forest model.
- **Best Random Forest Score**: R² score for the best Random Forest model after hyperparameter tuning.

## 🤝 Contributions
Contributions are welcome! Feel free to fork the repository and submit a pull request.

