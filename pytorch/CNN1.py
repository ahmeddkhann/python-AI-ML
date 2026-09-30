# ============================================================
# CONVOLUTIONAL NEURAL NETWORK (CNN) USING PYTORCH
# ============================================================

# Important Note:
# to test the model run the eval_cnn1.py file
# and to add your own custom images, add it in /pytorch/testdata/images
# you need to change the image path to test a specific image.


# In this program we will train a CNN on the CIFAR-10 dataset.
# CIFAR-10 contains 60,000 color images.
# Each image:
#     Height = 32 pixels
#     Width  = 32 pixels
#     Channels = 3 (RGB)
# Therefore one image has the shape:
#     [3, 32, 32]
# CIFAR-10 has 10 classes:
#     0 -> airplane
#     1 -> automobile
#     2 -> bird
#     3 -> cat
#     4 -> deer
#     5 -> dog
#     6 -> frog
#     7 -> horse
#     8 -> ship
#     9 -> truck


# ------------------------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------------------------

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# torchvision provides datasets and computer-vision utilities.
import torchvision

# transforms allows us to preprocess images before
# sending them into the neural network.
import torchvision.transforms as transforms


from torch.utils.data import DataLoader


# ============================================================
# IMAGE TRANSFORMATIONS
# ============================================================

transform = transforms.Compose([
    # ToTensor()
    # CIFAR-10 images are originally represented as images.
    # Neural networks work with tensors.
    # ToTensor() converts the image into a PyTorch tensor.
    # Original image:
    #
    #     Height = 32
    #     Width  = 32
    #     RGB    = 3 channels
    # Result:
    #     [3, 32, 32]
    # It also converts pixel values from approximately:
    #     0 - 255  into  0.0 - 1.0
    transforms.ToTensor(),

    # Normalize()
    # We normalize each RGB channel.
    # mean:  (0.5, 0.5, 0.5)
    # std: (0.5, 0.5, 0.5)
    # Formula:
    #     normalized = (pixel - mean) / std
    # For example:
    # pixel = 1.0
    # mean  = 0.5
    # std   = 0.5
    # normalized:
    #     (1.0 - 0.5) / 0.5
    #     = 1.0
    # And:
    #     pixel = 0.0
    #     (0.0 - 0.5) / 0.5
    #     = -1.0
    # So the values approximately become:
    #     0 -> -1
    #     1 -> +1
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ============================================================
# LOAD CIFAR-10 DATASET
# ============================================================

train_data = torchvision.datasets.CIFAR10(

    # Where should the dataset be stored?
    root="./data",

    # train=True means:
    # use the 50,000 training images.
    train=True,

    # Apply our transformations to every image.
    transform=transform,

    # Download the dataset if it isn't already present.
    download=True
)


test_data = torchvision.datasets.CIFAR10(

    root="./data",

    # train=False means:
    # use the 10,000 test images.
    train=False,

    transform=transform
)


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(

    train_data,
    
    # BATCH SIZE
    # Instead of giving all 50,000 images to the network
    # at once, we divide them into smaller groups.
    batch_size=32,

    # Randomize the order of training examples.
    # This helps prevent the model from learning based
    # on the ordering of the dataset.
    shuffle=True,

    # Number of worker processes used to load data.
    # 2 means PyTorch can use two worker processes
    # for loading batches.
    num_workers=0
)


test_loader = DataLoader(

    test_data,

    batch_size=32,

    # We normally don't need random ordering for testing.
    shuffle=False,

    num_workers=0
)


# ============================================================
# CNN MODEL
# ============================================================

class ConvulationNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # FIRST CONVOLUTIONAL LAYER
        self.conv1 = nn.Conv2d(

            # in_channels = 3
            # CIFAR-10 images are RGB.
            # RGB has:
            #     Red
            #     Green
            #     Blue
            # Therefore:
            #     input channels = 3
            in_channels=3,

            # ------------------------------------------------
            # out_channels = 12
            # ------------------------------------------------
            # We want 12 different filters.
            # Each filter can learn a different feature.
            # For example, filters may learn things such as:
            #     edges
            #     corners
            #     textures
            #     color patterns
            # These are learned automatically during training.
            out_channels=12,

            # kernel_size = 5
            # Each filter is 5 x 5 pixels spatially.
            # Since the input has 3 channels, each filter is
            # actually:
            #     3 x 5 x 5
            # = 75 weights
            # per filter, plus one bias.
           kernel_size=5
        )

        # MAX POOLING

        self.pool = nn.MaxPool2d(
            # Kernel size:
            # 2 x 2 region
           kernel_size= 2,

            # Stride:
            # move the 2x2 window by 2 pixels.
           stride= 2
        )


        # ====================================================
        # SECOND CONVOLUTIONAL LAYER
        # ====================================================

        self.conv2 = nn.Conv2d(

            # Input channels:
            # conv1 produced 12 feature maps.
            in_channels=12,

            # We now want 24 feature maps.
            out_channels=24,

            # 5x5 convolution kernel.
            kernel_size=5
        )

        # FULLY CONNECTED LAYER

        # We will calculate the size carefully below.
        # After our CNN operations the tensor will be:
        #     [batch_size, 24, 5, 5]
        # Therefore each image contains:
        #     24 * 5 * 5 = 600 values.
        # So our Linear layer expects 600 inputs.
        self.fc1 = nn.Linear(
            24 * 5 * 5, 
            # Output 120 neurons.
            120
        )


        # Second fully connected layer.
        # 120 inputs
        # 84 outputs
        self.fc2 = nn.Linear(120, 84)

        # Final layer.
        # CIFAR-10 has 10 classes.
        # Therefore:
        #     84 inputs
        #     10 outputs
        self.fc3 = nn.Linear(84, 10)


    # ========================================================
    # FORWARD PASS
    # ========================================================
    # This defines how data moves through our network.
    def forward(self, x):

        # FIRST CONVOLUTION + RELU + POOL


        x = self.conv1(x)

        # Apply ReLU.
        # Negative values become 0.
        x = F.relu(x)

        # Reduce spatial dimensions using max pooling.
        x = self.pool(x)


        # ----------------------------------------------------
        # SECOND CONVOLUTION + RELU + POOL
        # ----------------------------------------------------

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # FLATTEN
        # Before flatten:
        #     [batch_size, 24, 5, 5]
        # A Linear layer expects:
        #     [batch_size, features]
        # Therefore we need to convert:
        #     24 x 5 x 5
        # into:
        #     600
        # torch.flatten(x, 1)
        # means:
        # keep dimension 0 (batch dimension)
        # and flatten everything from dimension 1 onward.
        x = torch.flatten(x, 1)


        # ----------------------------------------------------
        # FULLY CONNECTED LAYERS
        # ----------------------------------------------------

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        # Final output.
        # No ReLU here.
        #
        # CrossEntropyLoss expects raw logits.
        x = self.fc3(x)

        return x


# CREATE MODEL
net = ConvulationNeuralNetwork()

# LOSS FUNCTION
loss_function = nn.CrossEntropyLoss()

# CrossEntropyLoss is commonly used for multi-class
# classification.
# We have 10 possible classes.
# The final layer produces:
#     [10 values]
# for each image.
# CrossEntropyLoss compares those values with the
# correct class label.

# OPTIMIZER
optimizer = optim.SGD(

    # Give SGD all trainable parameters of our network.
    net.parameters(),

    # LEARNING RATE
    # lr controls how large each weight update is.
    # Small learning rate:
    #     slower learning
    #     potentially more stable
    # Large learning rate:
    #     faster learning
    #     but may overshoot good solutions
    lr=0.001,

    # MOMENTUM
    # Momentum helps SGD continue moving in useful directions
    # instead of reacting only to the current gradient.
    # 0.9 is a common value.
    momentum=0.9
)

# NUMBER OF EPOCHS

epochs = 30

# An epoch means:
#     ONE COMPLETE PASS through the training dataset.
# CIFAR-10 has 50,000 training images.
# batch_size = 32
# Therefore approximately:
#     50,000 / 32
#     ≈ 1,563 batches per epoch.
# With 30 epochs:
#     ~46,890 training batches.


# TRAINING LOOP

for epoch in range(epochs):

    print(
        f"Training epoch... {epoch + 1}/{epochs}"
    )

    running_loss = 0.0


    # --------------------------------------------------------
    # ITERATE THROUGH TRAINING BATCHES
    # --------------------------------------------------------

    for i, data in enumerate(train_loader):

        # data contains: inputs labels
        inputs, labels = data
        
        # RESET GRADIENTS
        # PyTorch accumulates gradients by default.
        # We don't want gradients from the previous batch
        # to remain here.
        optimizer.zero_grad()

        # FORWARD PASS
        # Send images through CNN.
        outputs = net(inputs)

        # CALCULATE LOSS
        loss = loss_function(outputs, labels)

        # BACKPROPAGATION
        # Calculate gradients for all trainable parameters.
        loss.backward()

        # UPDATE PARAMETERS
        # SGD uses the calculated gradients to update
        # the weights.
        optimizer.step()

        # Store loss so we can calculate average loss later.
        running_loss += loss.item()


    # --------------------------------------------------------
    # PRINT AVERAGE LOSS
    # --------------------------------------------------------

    print(
        f"Loss: {running_loss / len(train_loader):.4f}"
    )

# SAVE TRAINED MODEL

torch.save(
    net.state_dict(),
    "cnn1.pth"
)


