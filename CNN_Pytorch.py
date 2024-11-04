import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Zvýšená velikost batch
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Další vylepšený model
class EnhancedNN_v3(nn.Module):
    def __init__(self, input_size):
        super(EnhancedNN_v3, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 2)  # Dvě třídy: Approved a Rejected
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

# Inicializace modelu, loss funkce a optimalizátoru s L2 regularizací
input_size = X_train.shape[1]
model = EnhancedNN_v3(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # L2 regularizace s weight_decay

# Použití scheduleru pro dynamické snížení learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Trénink modelu s opravou `running_loss`
epochs = 30  # Zkrácený počet epoch na 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()  # Přidání ztráty z každé dávky
    
    # Vypočítání průměrné ztráty za epochu
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


# Vyhodnocení na testovací sadě
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
