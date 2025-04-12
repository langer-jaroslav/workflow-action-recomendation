import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import random
from itertools import product

# ========== CONFIGURATION ==========
FILE_PATH = './data/requests.csv'
RANDOM_STATE = 42
N_COMBINATIONS = 20
EPOCHS = 30

# ========== DATA PREPARATION ==========
data = pd.read_csv(FILE_PATH)
data['status'] = data['status'].map({'Approved': 1, 'Rejected': 0})
data['urgency_vs_priority'] = data.apply(lambda row: int(row['is_urgent'] and row['priority'] == 'low'), axis=1)
data = data.drop(columns=['request_id', 'employee_id', 'risk_score_category'])
data = pd.get_dummies(data, drop_first=True)

X = data.drop('status', axis=1)
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)

# ========== HYPERPARAMETER SPACE ==========
hyper_space = {
    'lr': [1e-4, 3e-4, 1e-3],
    'batch_size': [32, 64, 128],
    'dropout': [0.3, 0.4, 0.5],
    'conv_filters': [32, 64, 128]
}

param_combinations = random.sample(list(product(*hyper_space.values())), N_COMBINATIONS)
best_acc = 0
best_f1 = 0
best_config = None
best_y_true = []
best_y_pred = []

# ========== MODEL TRAINING ==========
def build_model(input_shape, conv_filters, dropout_rate, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(conv_filters, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(conv_filters * 2, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

for i, (lr, batch_size, dropout, conv_filters) in enumerate(param_combinations):
    model = build_model(input_shape=(X_train.shape[1], 1), conv_filters=conv_filters, dropout_rate=dropout, learning_rate=lr)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=batch_size, verbose=0)

    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"[{i+1}/{N_COMBINATIONS}] acc={acc:.4f} f1={f1:.4f} | lr={lr}, batch={batch_size}, dropout={dropout}, filters={conv_filters}")

    if acc > best_acc:
        best_acc = acc
        best_f1 = f1
        best_config = (lr, batch_size, dropout, conv_filters)
        best_y_true = y_test
        best_y_pred = y_pred

# ========== FINAL BEST ==========
print("\n--TensorFlow CNN")
print("    Best Model Hyperparameters:")
print(f"      learning_rate: {best_config[0]}")
print(f"      batch_size: {best_config[1]}")
print(f"      dropout: {best_config[2]}")
print(f"      conv_filters: {best_config[3]}")

print("\n    Main Parameters Used:")
print(f"      File path: {FILE_PATH}")
print(f"      Train size: {len(X_train)}")
print(f"      Test size: {len(X_test)}")
print(f"      Random state: {RANDOM_STATE}")
print(f"      Number of parameter combinations: {N_COMBINATIONS}")
print(f"      Epochs: {EPOCHS}")

print(f"\n    Best test set accuracy: {best_acc:.3f}")
print("\n    Classification Report:")
print(classification_report(best_y_true, best_y_pred, digits=3))
