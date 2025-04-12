# Randomized hyperparameter search for PyTorch CNN

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import shuffle
from itertools import product
import random

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

X_train = X_train.reshape(-1, 1, X_train.shape[1])
X_test = X_test.reshape(-1, 1, X_test.shape[1])

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TabularDataset(X_train, y_train)
test_dataset = TabularDataset(X_test, y_test)

# ========== CNN DEFINITION ==========
class CNNTabular(nn.Module):
    def __init__(self, input_features, conv_filters, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_filters),
            nn.LeakyReLU(),
            nn.Conv1d(conv_filters, conv_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_filters * 2),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(conv_filters * 2, 64),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()

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

# ========== TRAINING LOOP ==========
def train_one(model, loader, optimizer, criterion):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

for i, (lr, batch_size, dropout, conv_filters) in enumerate(param_combinations):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = CNNTabular(X_train.shape[2], conv_filters, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):
        train_one(model, train_loader, optimizer, criterion)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            y_true.extend(y_batch.int().numpy())
            y_pred.extend((preds >= 0.5).int().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"[{i+1}/{N_COMBINATIONS}] acc={acc:.4f} f1={f1:.4f} | lr={lr}, batch={batch_size}, dropout={dropout}, filters={conv_filters}")

    if acc > best_acc:
        best_acc = acc
        best_f1 = f1
        best_config = (lr, batch_size, dropout, conv_filters)
        best_y_true = y_true
        best_y_pred = y_pred

# ========== FINAL BEST ==========
print("\n--PyTorch CNN")
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
