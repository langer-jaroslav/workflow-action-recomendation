import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Parameters
file_path = './data/requests.csv'
test_size = 0.3
random_state = 42

# Load and preprocess data
data = pd.read_csv(file_path)
data['status'] = data['status'].apply(lambda x: 1 if x == 'Approved' else 0)
X = pd.get_dummies(data.drop('status', axis=1), drop_first=True)
y = data['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Hyperparameter distribution
param_dist = {
    'n_estimators': np.arange(50, 401, 50),
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

# Randomized search
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=random_state),
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=5,
    random_state=random_state,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)
best_params_xgb = random_search.best_params_
best_score_xgb = random_search.best_score_

# Train and evaluate the model
best_xgb_model = XGBClassifier(**best_params_xgb, eval_metric='logloss', random_state=random_state)
best_xgb_model.fit(X_train, y_train)
y_pred_xgb = best_xgb_model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
classification_rep_xgb = classification_report(y_test, y_pred_xgb)

# Output results
print("\nBest Hyperparameters from RandomizedSearchCV:")
for param, value in best_params_xgb.items():
    print(f"  {param}: {value}")

print("\nMain Parameters Used:")
print(f"  File path: {file_path}")
print(f"  Test size: {test_size}")
print(f"  Random state: {random_state}")
print(f"  Number of iterations for RandomizedSearchCV: 50")
print(f"  Cross-validation folds: 5")

print(f"\nBest cross-validation accuracy: {best_score_xgb:.3f}")
print(f"Test set accuracy: {accuracy_xgb:.3f}")
print("\nClassification Report:\n", classification_rep_xgb)
