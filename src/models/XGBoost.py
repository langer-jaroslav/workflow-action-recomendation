import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ========== GLOBAL CONFIGURATION ==========
FILE_PATH = './data/requests.csv'
RANDOM_STATE = 42
N_COMBINATIONS = 200
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25  # 25% of 80% = 20%

# ========== LOAD & PREPROCESS DATA ==========
data = pd.read_csv(FILE_PATH)

data['status'] = data['status'].map({'Approved': 1, 'Rejected': 0})

data['urgency_vs_priority'] = data.apply(
    lambda row: int(row['is_urgent'] and row['priority'] == 'low'),
    axis=1
)

data = data.drop(columns=['request_id', 'employee_id', 'risk_score_category'])

data = pd.get_dummies(data, drop_first=True)

X = data.drop('status', axis=1)
y = data['status']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y_temp
)

# ========== DEFINE HYPERPARAMETER SPACE ==========
param_dist = {
    "n_estimators": [100, 300, 500, 700, 1000],
    "max_depth": [5, 7, 9, 10, 12],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "reg_lambda": [0, 1, 10, 50],
    "reg_alpha": [0, 0.1, 1, 10],
    "scale_pos_weight": [1, 1.5, 2, 3],
    "gamma": [0, 0.1, 0.3, 1, 5],              
    "min_child_weight": [1, 2, 5, 10],            
    "max_delta_step": [0, 1, 5]            
}

param_list = list(ParameterSampler(param_dist, n_iter=N_COMBINATIONS, random_state=RANDOM_STATE))

# ========== TRAINING LOOP ==========
best_model = None
best_params = None
best_val_acc = 0
best_test_acc = 0
best_y_pred = None

print(f"\n🔬 Testing {N_COMBINATIONS} parameter combinations...\n")

for i, params in enumerate(param_list):
    model = XGBClassifier(
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        **params
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = accuracy_score(y_test, model.predict(X_test))
        best_model = model
        best_params = params
        best_y_pred = model.predict(X_test)

    print(f"[{i+1:02}/{N_COMBINATIONS}] val_acc={val_acc:.4f}  best_so_far={best_val_acc:.4f}")

# ========== FINAL OUTPUT ==========
print("\n\n🔥 Best Model Hyperparameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

print(f"""
Main Parameters Used:
  File path: {FILE_PATH}
  Train size: {len(X_train)}
  Validation size: {len(X_val)}
  Test size: {len(X_test)}
  Random state: {RANDOM_STATE}
  Number of parameter combinations: {N_COMBINATIONS}
""")

print(f"Best validation accuracy: {best_val_acc:.3f}")
print(f"Test set accuracy: {best_test_acc:.3f}\n")

print("Classification Report:")
print(classification_report(y_test, best_y_pred, digits=3))