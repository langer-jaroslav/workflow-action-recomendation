import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

# Kontrola dostupnosti GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Načtení dat
data = pd.read_csv('./data/requests.csv')

# Zpracování a příprava dat
label_encoder = LabelEncoder()
data['status'] = label_encoder.fit_transform(data['status'])  # 1 = Approved, 0 = Rejected
data['priority'] = label_encoder.fit_transform(data['priority'])
data['order_type'] = label_encoder.fit_transform(data['order_type'])
data['is_urgent'] = data['is_urgent'].astype(int)
data['is_from_wholesaler'] = data['is_from_wholesaler'].astype(int)

# Oddělení cíle a vstupů
X = data.drop(['request_id', 'status'], axis=1).values
y = data['status'].values

# Normalizace vstupních dat
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Rozdělení na trénovací a testovací sadu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definice Datasetu pro PyTorch
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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Vylepšený model
class EnhancedNN(nn.Module):
    def __init__(self, input_size):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)  # Dvě třídy: Approved a Rejected
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Inicializace modelu, loss funkce a optimalizátoru
input_size = X_train.shape[1]
model = EnhancedNN(input_size).to(device)  # Přesun modelu na GPU, pokud je k dispozici
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005)

# Trénink modelu s vylepšením
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Přesun dat na GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Vyhodnocení na testovací sadě
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Přesun testovacích dat na GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
