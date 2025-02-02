import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import itertools

# 🔥 Load dataset
file_path = "./data/requests.csv"
data = pd.read_csv(file_path)

# ❌ Remove `request_id` if it exists
if "request_id" in data.columns:
    data = data.drop(columns=["request_id"])
    print("✅ `request_id` removed.")

# 🎯 Encode target variable ('status') as binary (1 = Approved, 0 = Rejected)
data['status'] = data['status'].apply(lambda x: 1 if x == 'Approved' else 0)

# 🛠️ One-hot encoding for categorical variables
X = pd.get_dummies(data.drop(columns=["status"]), drop_first=True)
y = data['status'].values

# 📊 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 📏 Normalize features (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔄 Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 🔄 Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class CNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(CNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x

    

# Možné hodnoty hyperparametrů
lr_values = [0.0001, 0.001, 0.005]
batch_sizes = [16, 32, 64]
dropout_values = [0.3, 0.4, 0.5]
hidden_sizes = [128, 256, 512]
weight_decay_values = [0, 1e-4, 1e-3]

# Všechny kombinace hyperparametrů
hyperparam_combinations = list(itertools.product(lr_values, batch_sizes, dropout_values, hidden_sizes, weight_decay_values))

best_acc = 0
best_params = None

for lr, batch_size, dropout, hidden_size, weight_decay in hyperparam_combinations:
    print(f"🔍 Testing: lr={lr}, batch_size={batch_size}, dropout={dropout}, hidden_size={hidden_size}, weight_decay={weight_decay}")

    # 🔥 Definice modelu
    model = CNNClassifier(input_size=X_train.shape[1], hidden_size=hidden_size, dropout=dropout)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ⚡ Rychlý testovací trénink (10 epoch)
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # 🚀 Evaluace
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            y_test_pred = model(batch_X)
            y_pred_list.extend(y_test_pred.squeeze().tolist())

    # 📊 Výpočet metrik
    y_pred_bin = [1 if x >= 0.5 else 0 for x in y_pred_list]
    accuracy = accuracy_score(y_test, y_pred_bin)

    print(f"🎯 Accuracy: {accuracy:.3f}")

    # 🎯 Uložit nejlepší hyperparametry
    if accuracy > best_acc:
        best_acc = accuracy
        best_params = (lr, batch_size, dropout, hidden_size, weight_decay)

print(f"\n✅ Best Model: lr={best_params[0]}, batch_size={best_params[1]}, dropout={best_params[2]}, hidden_size={best_params[3]}, weight_decay={best_params[4]}")

# 🎯 Použití nejlepších hyperparametrů
lr, batch_size, dropout, hidden_size, weight_decay = best_params

model = CNNClassifier(input_size=X_train.shape[1], hidden_size=hidden_size, dropout=dropout)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 🏋️‍♂️ Finální trénink (50 epoch)
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # 🚀 Evaluation
model.eval()
y_pred_list = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        y_test_pred = model(batch_X)
        y_pred_list.extend(y_test_pred.squeeze().tolist())


# 📊 Výpočet metrik
y_pred_bin = [1 if x >= 0.5 else 0 for x in y_pred_list]
accuracy = accuracy_score(y_test, y_pred_bin)
class_report = classification_report(y_test, y_pred_bin)

print(f"\n--CNN PyTorch Model")
print(f"    Test set accuracy: {accuracy:.3f}\n")
print("    Classification Report:")
print(class_report)