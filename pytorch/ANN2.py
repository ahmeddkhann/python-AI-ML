import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

x, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
y = y.astype("int64")
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                         train_size=0.8,random_state=40)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

pca = PCA(n_components=100)
x_train_reduced = pca.fit_transform(x_train_scaled)
x_test_reduced = pca.transform(x_test_scaled)

x_train_reduced_tensor = torch.from_numpy(x_train_reduced).float()
x_test_reduced_tensor = torch.from_numpy(x_test_reduced).float()

y_train_tensor = torch.from_numpy(y_train).long()
y_test_tensor = torch.from_numpy(y_test).long()

print(x_train_reduced_tensor.shape)

train_dataset = TensorDataset(x_train_reduced_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


class OpenMlMnist(nn.Module):
    def __init__(self):
        super(OpenMlMnist, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10 )
        
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x
    
model = OpenMlMnist()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
epochs = 20

for epoch in range (epochs):
    model.train()
    running_loss = 0.0
    
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = loss_function(preds, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Epoch {epoch + 1}: Loss was: {running_loss}")

with torch.no_grad():
    model.eval()
    preds = model(x_test_reduced_tensor)
    loss = loss_function(preds, y_test_tensor).item()
    probabilities = F.softmax(preds, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    correct = (predictions == y_test_tensor).sum().item()
    accuracy = correct / len(y_test_tensor)

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2%}")
    
    # Print first 5 predictions
    for i in range(5):
        print(f"\nImage {i + 1}")
        print("Actual:", y_test_tensor[i].item())
        print("Predicted:", predictions[i].item())
        print("Probabilities:", probabilities[i])
    