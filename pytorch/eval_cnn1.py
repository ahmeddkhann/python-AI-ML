import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ============================================================
# 1. CIFAR-10 CLASS NAMES
# ============================================================

# The output of our neural network contains 10 numbers.
# Each position corresponds to one of these classes.

# index 0 -> plane
# index 1 -> car
# index 2 -> bird
# index 3 -> cat
# index 4 -> deer
# index 5 -> dog
# index 6 -> frog
# index 7 -> horse
# index 8 -> ship
# index 9 -> truck

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


# ============================================================
# 2. DEFINE THE SAME CNN ARCHITECTURE
# ============================================================

class ConvulationNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=12, kernel_size=5 )

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # Second convolution
        self.conv2 = nn.Conv2d(in_channels=12,out_channels=24,kernel_size=5 )

        # Fully connected layers
        self.fc1 = nn.Linear(24 * 5 * 5,120 )

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84,10 )


    def forward(self, x):

        # Convolution → ReLU → Pool
        x = self.pool(F.relu(self.conv1(x)))

        # Convolution → ReLU → Pool
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten CNN output
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Final output
        x = self.fc3(x)

        return x


# ============================================================
# 3. CREATE THE MODEL
# ============================================================

model = ConvulationNeuralNetwork()


# ============================================================
# 4. LOAD THE TRAINED PARAMETERS
# ============================================================

model.load_state_dict(
    torch.load(
        "cnn1.pth",
        map_location=torch.device("cpu")
    )
)


# ============================================================
# 5. PUT MODEL INTO EVALUATION MODE
# ============================================================

model.eval()


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ============================================================
# 7. LOAD IMAGE
# ============================================================

image = Image.open(
    "testdata/images/aeroplane.jpeg"
).convert("RGB")


# ============================================================
# 8. APPLY TRANSFORM
# ============================================================

image = transform(image)


# ============================================================
# 9. ADD BATCH DIMENSION
# ============================================================

image = image.unsqueeze(0)


# ============================================================
# 10. RUN MODEL
# ============================================================

with torch.no_grad():

    outputs = model(image)


# ============================================================
# 11. GET PREDICTED CLASS
# ============================================================

_, predicted = torch.max(outputs, 1)


# ============================================================
# 12. PRINT RESULT
# ============================================================

predicted_class = classes[predicted.item()]

print("Predicted class:", predicted_class)