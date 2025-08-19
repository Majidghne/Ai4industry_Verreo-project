import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# **1. Transformations et Chargement des Données**
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Redimensionne les images
    transforms.ToTensor(),        # Convertit en tenseur
    transforms.Normalize((0.5,), (0.5,))  # Normalise les valeurs de pixel
])

# Chargement des données
train_data = datasets.ImageFolder('data/train', transform=transform)
val_data = datasets.ImageFolder('data/val', transform=transform)
test_data = datasets.ImageFolder('data/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Nombre de classes détectées
num_classes = len(train_data.classes)
print(f"Classes détectées : {train_data.classes}")

# **2. Création du Réseau de Neurones**
class NeuralNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Couche dense avec 64 neurones
        self.fc2 = nn.Linear(64, 64)        # Couche dense avec 64 neurones
        self.fc3 = nn.Linear(64, num_classes)  # Couche de sortie

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Aplatit les images
        x = torch.relu(self.fc1(x))        # Activation ReLU
        x = torch.relu(self.fc2(x))        # Activation ReLU
        x = self.fc3(x)                    # Sortie (scores bruts)
        return x

# Initialisation du modèle
input_dim = 64 * 64 * 3  # Images RGB redimensionnées en 64x64
model = NeuralNet(input_dim=input_dim, num_classes=num_classes)

# **3. Fonction de perte et Optimiseur**
criterion = nn.CrossEntropyLoss()  # Fonction de perte pour la classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimiseur Adam

# **4. Entraînement du Modèle**
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Mode entraînement
    running_loss = 0.0

    for inputs, labels in train_loader:
        # Remise à zéro des gradients
        optimizer.zero_grad()
        # Passage avant
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Rétropropagation
        loss.backward()
        # Mise à jour des poids
        optimizer.step()

        running_loss += loss.item()

    print(f"Époque {epoch+1}/{num_epochs}, Perte : {running_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Précision sur les données de validation : {100 * correct / total:.2f}%")

# **5. Évaluation sur le Test**
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Précision sur les données de test : {100 * correct / total:.2f}%")
