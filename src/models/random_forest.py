import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
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

# 📌 Hyperparameter tuning - defining search space
param_dist = {
    'n_estimators': np.arange(1000, 2001, 200),  # Number of trees
    'max_depth': [None, 30, 40, 50],  # Maximum depth of trees
    'min_samples_split': [5, 10, 20],  # Minimum number of samples to split a node
    'min_samples_leaf': [2, 4, 6],  # Minimum number of samples per leaf node
    'bootstrap': [True, False],  # Whether to use bootstrapping
    'class_weight': ['balanced']  # Balancing class weights
}

# RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=random_state),
    param_distributions=param_dist,
    n_iter=50,  # Number of random hyperparameter combinations to try
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    random_state=random_state,
    n_jobs=-1,
    verbose=1
)

# **Run hyperparameter tuning**
random_search.fit(X_train, y_train)

# Retrieve the best hyperparameters
best_params_rf = random_search.best_params_
best_score_rf = random_search.best_score_

# **Train the Random Forest model with best hyperparameters**
best_rf_model = RandomForestClassifier(**best_params_rf, random_state=random_state)
best_rf_model.fit(X_train, y_train)

# **Predictions on test data**
y_pred_rf = best_rf_model.predict(X_test)

# **Compute accuracy and classification report**
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)

# 📌 Print formatted results
print("\n--Random Forest")
print("    Best Model Hyperparameters from RandomizedSearchCV:")
for param, value in best_params_rf.items():
    print(f"      {param}: {value}")

print("\n    Main Parameters Used:")
print(f"      File path: {file_path}")
print(f"      Test size: {test_size}")
print(f"      Random state: {random_state}")
print(f"      Number of iterations for RandomizedSearchCV: 50")  # Počet testovaných kombinací hyperparametrů
print(f"      Cross-validation folds: 5")  # Počet částí (foldů) pro křížovou validaci

print(f"\n    Best cross-validation accuracy: {best_score_rf:.3f}")
print(f"    Test set accuracy: {accuracy_rf:.3f}\n")

print("    Classification Report:")
print(classification_rep_rf)
