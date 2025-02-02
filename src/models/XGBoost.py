import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

# Parameters
file_path = "./data/requests.csv"
test_size = 0.3
random_state = 42

# Load dataset
data = pd.read_csv(file_path)

# ❌ Remove `request_id` if it exists
if "request_id" in data.columns:
    data = data.drop(columns=["request_id"])
    print("✅ `request_id` removed.")

# 🔀 Shuffle dataset to ensure randomness
data = shuffle(data, random_state=random_state)

# 📌 Feature Engineering - creating new features
data['log_total_value'] = np.log1p(data['total_value'])  # Log transformation of total value
data['wholesale_urgency'] = data['is_from_wholesaler'] * data['is_urgent']  # Interaction of wholesale and urgency
data['sqrt_total_value'] = np.sqrt(data['total_value'])  # Square root transformation
data['total_urgent'] = data['total_value'] * data['is_urgent']  # Interaction feature
data['price_age_ratio'] = data['price_per_item'] / (data['request_age'] + 1)  # Avoid division by zero
data['risk_bin'] = pd.qcut(data['risk_score'], q=4, labels=[1, 2, 3, 4])  # Binning risk_score into categories
# ❌ Drop weak features
columns_to_drop = ["requested_items", "total_value", "price_per_item", "order_type_other"]
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Encode target variable ('status') as binary (1 = Approved, 0 = Rejected)
data['status'] = data['status'].apply(lambda x: 1 if x == 'Approved' else 0)

# One-hot encoding for categorical variables
X = pd.get_dummies(data.drop(columns=["status"]), drop_first=True)
y = data['status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# 📌 Hyperparameter tuning - defining search space for XGBoost
param_dist_xgb = {
    'n_estimators': np.arange(400, 1001, 200),  # Number of boosting rounds
    'max_depth': [5, 7, 10],  # Maximum tree depth
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate
    'subsample': [0.7, 0.8, 0.9],  # Subsample ratio of training instances
    'colsample_bytree': [0.7, 0.8, 0.9],  # Subsample ratio of columns when constructing each tree
    'gamma': [0.1, 0.2, 0.3],  # Minimum loss reduction for split
    'reg_lambda': [1, 5, 10],  # L2 regularization
    'reg_alpha': [0, 1, 3],  # L1 regularization
}

# RandomizedSearchCV to find the best hyperparameters
random_search_xgb = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
    param_distributions=param_dist_xgb,
    n_iter=50,  # Number of random hyperparameter combinations to try
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    random_state=random_state,
    n_jobs=-1,
    verbose=1
)

# **Run hyperparameter tuning**
random_search_xgb.fit(X_train, y_train)

# Retrieve the best hyperparameters
best_params_xgb = random_search_xgb.best_params_
best_score_xgb = random_search_xgb.best_score_

# **Train the XGBoost model with best hyperparameters**
best_xgb_model = xgb.XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='logloss', random_state=random_state)
best_xgb_model.fit(X_train, y_train)

# **Predictions on test data**
y_pred_xgb = best_xgb_model.predict(X_test)

# **Compute accuracy and classification report**
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
classification_rep_xgb = classification_report(y_test, y_pred_xgb)

# 📌 Print formatted results
print("\n--XGBoost")
print("    Best Model Hyperparameters from RandomizedSearchCV:")
for param, value in best_params_xgb.items():
    print(f"      {param}: {value}")

print("\n    Main Parameters Used:")
print(f"      File path: {file_path}")
print(f"      Test size: {test_size}")
print(f"      Random state: {random_state}")
print(f"      Number of iterations for RandomizedSearchCV: 50")
print(f"      Cross-validation folds: 5")

print(f"\n    Best cross-validation accuracy: {best_score_xgb:.3f}")
print(f"    Test set accuracy: {accuracy_xgb:.3f}\n")

print("    Classification Report:")
print(classification_rep_xgb)
