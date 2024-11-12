import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Parameters
file_path = './data/requests.csv'
test_size = 0.3
random_state = 42

# Load data
data = pd.read_csv(file_path)

# Encode target variable ('status') as binary (1 = Approved, 0 = Rejected)
data['status'] = data['status'].apply(lambda x: 1 if x == 'Approved' else 0)
X = data.drop('status', axis=1)
y = data['status']

# One-hot encoding for categorical features
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Define hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(100, 1001, 100),  # Values from 100 to 1000 with step 100
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'bootstrap': [True, False]
}

# Setup RandomizedSearchCV with Random Forest and the parameter distribution
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=random_state),
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=5,
    random_state=random_state,
    n_jobs=-1,
    verbose=1
)

# Perform the search
random_search.fit(X_train, y_train)

# Get best parameters and best score from search
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

# Train the Random Forest model with the best parameters found
best_rf_model_random = RandomForestClassifier(**best_params_random, random_state=random_state)
best_rf_model_random.fit(X_train, y_train)

# Predict on test data and generate metrics
y_pred_random = best_rf_model_random.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)
classification_rep_random = classification_report(y_test, y_pred_random)

# Display results
print("\nBest Model Hyperparameters from RandomizedSearchCV:")
for param, value in best_params_random.items():
    print(f"  {param}: {value}")

# Display main parameters for consistency
print("\nMain Parameters Used:")
print(f"  File path: {file_path}")
print(f"  Test size: {test_size}")
print(f"  Random state: {random_state}")
print(f"  Number of iterations for RandomizedSearchCV: 50")
print(f"  Cross-validation folds: 5")

print(f"\nBest cross-validation accuracy: {best_score_random:.3f}")
print(f"Test set accuracy: {accuracy_random:.3f}")
print("\nClassification Report:\n", classification_rep_random)
