import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score

# ========== GLOBAL CONFIGURATION ==========
FILE_PATH = './data/requests.csv'
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_ITER_SEARCH = 50
CV_FOLDS = 5

# ========== DATA LOADING & PREPROCESSING ==========
data = pd.read_csv(FILE_PATH)

data['status'] = data['status'].map({'Approved': 1, 'Rejected': 0})

data = data.drop(columns=['request_id', 'employee_id', 'risk_score_category'])

data = pd.get_dummies(data, drop_first=True)

X = data.drop('status', axis=1)
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ========== HYPERPARAMETER SEARCH SPACE ==========
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(400, 800, num=10)],
    'max_depth': [8, 10, 12, 15, 20],
    'min_samples_split': [2, 3, 4, 5, 6],
    'min_samples_leaf': [1, 2, 3],
    'bootstrap': [True],
    'class_weight': ['balanced'],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=RANDOM_STATE)

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=N_ITER_SEARCH,
    cv=CV_FOLDS,
    verbose=2,  
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring='accuracy'
)

# ========== MODEL TRAINING ==========
rf_random.fit(X_train, y_train)

best_model = rf_random.best_estimator_

y_pred = best_model.predict(X_test)

# ========== OUTPUT REPORT ==========
print("\nBest Model Hyperparameters from RandomizedSearchCV:")
for param, value in rf_random.best_params_.items():
    print(f"  {param}: {value}")

print(f"""
Main Parameters Used:
  File path: {FILE_PATH}
  Test size: {TEST_SIZE}
  Random state: {RANDOM_STATE}
  Number of iterations for RandomizedSearchCV: {N_ITER_SEARCH}
  Cross-validation folds: {CV_FOLDS}
""")

print(f"Best cross-validation accuracy: {rf_random.best_score_:.3f}")
print(f"Test set accuracy: {accuracy_score(y_test, y_pred):.3f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, digits=3))
