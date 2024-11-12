import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

file_path = './data/requests.csv'
test_size = 0.3
random_state = 42

data = pd.read_csv(file_path)

# Konverze cílové proměnné 'status' na binární hodnoty (1 pro Approved, 0 pro Rejected)
data['status'] = data['status'].apply(lambda x: 1 if x == 'Approved' else 0)

X = data.drop('status', axis=1)
y = data['status']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

param_dist = {
    'n_estimators': np.arange(100, 1001, 100),  # Hodnoty od 100 do 1000 s krokem 100
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'bootstrap': [True, False]
}

# Nastavení RandomizedSearchCV s modelem Random Forest a rozšířeným rozsahem parametrů
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

random_search.fit(X_train, y_train)

best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

best_rf_model_random = RandomForestClassifier(**best_params_random, random_state=random_state)
best_rf_model_random.fit(X_train, y_train)

y_pred_random = best_rf_model_random.predict(X_test)

accuracy_random = accuracy_score(y_test, y_pred_random)
classification_rep_random = classification_report(y_test, y_pred_random)

print("Nejlepší parametry:", best_params_random)
print("Nejlepší průměrná přesnost během cross-validace:", best_score_random)
print("\nPřesnost na testovací sadě:", accuracy_random)
print("\nClassification Report:\n", classification_rep_random)
