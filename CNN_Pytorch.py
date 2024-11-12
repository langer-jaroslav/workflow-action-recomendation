import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
random_state = 42
torch.manual_seed(random_state)
np.random.seed(random_state)

# Hyperparameters and configurations
file_path = './data/requests.csv'
test_size = 0.2
batch_size = 32
learning_rate = 0.001
weight_decay = 0.01
epochs = 30
step_size = 10
gamma = 0.5

# Load data
data = pd.read_csv(file_path)

# Data preprocessing
label_encoder = LabelEncoder()
data['status'] = label_encoder.fit_transform(data['status'])  # 1 = Approved, 0 = Rejected
data['priority'] = label_encoder.fit_transform(data['priority'])
data['order_type'] = label_encoder.fit_transform(data['order_type'])
data['is_urgent'] = data['is_urgent'].astype(int)
data['is_from_wholesaler'] = data['is_from_wholesaler'].astype(int)

# Split features and target
X = data.drop(['request_id', 'status'], axis=1).values
y = data['status'].values

# Normalize feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Define Dataset class for PyTorch
class RequestDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = RequestDataset(X_train, y_train)
test_dataset = RequestDataset(X_test, y_test)

# Data loaders for training and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model
class EnhancedNN_v3(nn.Module):
    def __init__(self, input_size):
        super(EnhancedNN_v3, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 2)  # Output layer for 2 classes
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]
model = EnhancedNN_v3(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Training loop with average loss calculation per epoch
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

# Evaluation on test set with classification report and accuracy
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Generate and display results in desired format
report = classification_report(y_true, y_pred, target_names=['Rejected', 'Approved'])
accuracy = accuracy_score(y_true, y_pred)

print("\nModel Hyperparameters:")
print(f"  File path: {file_path}")
print(f"  Test size: {test_size}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Weight decay: {weight_decay}")
print(f"  Epochs: {epochs}")
print(f"  Step size for LR scheduler: {step_size}")
print(f"  Gamma for LR scheduler: {gamma}")

print(f"\nTest set accuracy: {accuracy:.3f}")
print("\nClassification Report:\n", report)