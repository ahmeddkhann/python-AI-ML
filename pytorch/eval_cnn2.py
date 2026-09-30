import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os


classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)


# structuring over Neural Network Model that we created in CNN2.py

class EfficientConvulationNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
           in_channels= 64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
           in_channels= 128,
           out_channels=256,
            kernel_size=3,
            padding=1
        )

        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            256,
            10
        )

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = F.relu(self.bn4(self.conv4(x)))

        x = self.global_pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x



model =EfficientConvulationNeuralNetwork()

# loading the learned parameteres.
model.load_state_dict(
    torch.load(
        "cnn3.pth",
        map_location=torch.device("cpu")
    )
)

# turning the model in to the evaluation 
model.eval()


# transforming size and converting it to tensor + normalizing it.
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# loading all of the images in testdata/images directory
image_directory = "testdata/images"

image_files = [
    file
    for file in os.listdir(image_directory)
    if file.lower().endswith(".jpeg")
]


# joining the image path
for image_file in image_files:

    image_path = os.path.join(
        image_directory,
        image_file
    )
    
    # converting the image into the RGB form.
    image = Image.open(image_path).convert("RGB")
    
    # transforming the image into the tensor.
    image = transform(image)

    image = image.unsqueeze(0)
    
    # testing the model with our own images by letting it to predict
    # all of the images present in our directory.
    with torch.no_grad():
        outputs = model(image)

    _, predicted = torch.max(outputs, 1)

    predicted_class = classes[predicted.item()]

    print(
        f"{image_file} -> {predicted_class}"
    )