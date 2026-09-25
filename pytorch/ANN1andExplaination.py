# ============================================================
# IMPORTS
# ============================================================
# we will be building our first Neural Network using PyTorch.
import torch

# importing neural networks as nn from the torch library.
# nn contains classes that help us build neural network layers
import torch.nn as nn

# importing activation functions as F from functional neural
# networks. We can use functions such as relu() and sigmoid()
# from F.
import torch.nn.functional as F

# importing optimizers as optim.
# Optimizers are responsible for updating the weights of our
# neural network based on the gradients calculated during
# backpropagation.
import torch.optim as optim

# importing DataLoader and TensorDataset from utils.data.
# TensorDataset is used to combine our input features (x) and
# target values (y) into a dataset that PyTorch can work with.
# DataLoader is used to load this dataset into the model in
# smaller batches instead of sending the entire dataset at once.
from torch.utils.data import DataLoader, TensorDataset

# importing the breast cancer dataset from scikit-learn.
# This dataset contains 30 numerical features for each patient
# and a target value representing the class.
from sklearn.datasets import load_breast_cancer

# importing train_test_split which will split our dataset into
# training and testing portions.
from sklearn.model_selection import train_test_split

# importing StandardScaler which is used to standardize our
# numerical features so that they have a similar scale.
from sklearn.preprocessing import StandardScaler


# ============================================================
# LOADING AND PREPARING THE DATASET
# ============================================================

# loading the data as x and y from the dataset.
# x contains our input features.
# y contains the target/class that we want our neural network
# to predict.
# return_X_y=True tells sklearn to directly return the features
# and target instead of returning the complete dataset object.
x, y = load_breast_cancer(return_X_y=True)


# splitting the data into:
# x_train -> training input features
# x_test  -> testing input features
# y_train -> training target values
# y_test  -> testing target values
# The training data will be used to teach the neural network.
# The testing data will be kept separate and used later to see
# how well the trained model performs on unseen data.
# random_state=42 ensures that we get the same split every time
# we run the program.
# test_size=0.2 means:
# 80% of the data -> training
# 20% of the data -> testing/evaluation
x_train, x_test, y_train, y_test = train_test_split(  x,  y,
                              random_state=42, test_size=0.2 )


# ------------------------------------------------------------
# FEATURE SCALING
# ------------------------------------------------------------

# here we are scaling the input features.
# Our dataset contains different features that can have very
# different numerical ranges.
# For example, one feature might contain values around 0.1,
# while another feature might contain values around 1000.
# Neural networks generally train better when numerical features
# are on a similar scale.
# StandardScaler standardizes each feature approximately to:
# mean = 0
# standard deviation = 1
# We use fit_transform() ONLY on the training data because the
# scaler must learn its mean and standard deviation from the
# training data.
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# For the test data, we use transform() instead of fit_transform().
# This is important because the test data is supposed to represent
# unseen data. We must not allow information from the test set to
# influence the scaling process.
# Therefore:
# training data -> fit_transform()
# testing data  -> transform()
x_test_scaled = scaler.transform(x_test)


# ------------------------------------------------------------
# CONVERTING NUMPY ARRAYS INTO PYTORCH TENSORS
# ------------------------------------------------------------

# PyTorch neural networks work with PyTorch tensors.
# sklearn gives us NumPy arrays, so we convert our NumPy arrays
# into PyTorch tensors using torch.from_numpy().
# .float() converts the tensors to 32-bit floating point values
# (float32).
# Neural network calculations in PyTorch normally use floating
# point numbers because weights, activations, gradients and other
# mathematical operations require decimal values.
# For example:
# 1 -> 1.0
# 5 -> 5.0
# Using float32 is also the standard numerical type used by many
# neural network operations and is more efficient than using
# float64 in many cases.
x_train_scaled_tensor = torch.from_numpy(x_train_scaled).float()
x_test_scaled_tensor = torch.from_numpy(x_test_scaled).float()

# converting our target values from NumPy arrays into PyTorch
# tensors and then converting them to float32.
# We use float here because later we are going to use BCELoss
# (Binary Cross Entropy Loss), which expects the predictions and
# target values to be floating-point values.
# unsqueeze(1) adds an extra dimension to our target tensor.
# Before unsqueeze:
# y_train_tensor.shape
# (455,)
# After unsqueeze(1):
# y_train_tensor.shape
# (455, 1)
# Why do we need this?
# Our final neural network layer has only one output neuron:
# [prediction]
# Therefore, the model produces predictions with shape:
# (batch_size, 1)
# We want our target tensor to have the same shape so that the
# loss function can compare them directly.
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)


# ------------------------------------------------------------
# CREATING THE PYTORCH DATASET
# ------------------------------------------------------------

# TensorDataset combines our input features and target values
# into one dataset.
# This makes it easier for DataLoader to retrieve matching
# features and labels together.
train_dataset = TensorDataset( x_train_scaled_tensor, y_train_tensor)


# ------------------------------------------------------------
# CREATING THE DATALOADER
# ------------------------------------------------------------

# DataLoader is responsible for giving our training data to the
# neural network in batches.
# batch_size=32 means that the model will process up to 32
# training samples at one time.
# For example, if we have:
# 455 training samples
# batch_size = 32
# the DataLoader will divide those 455 samples into multiple
# batches.
# The model processes one batch, calculates the loss, performs
# backpropagation and updates its weights, then moves to the
# next batch.
# shuffle=True means that the training samples are randomly
# shuffled at the beginning of each epoch.
# This prevents the model from always seeing the training
# examples in exactly the same order.
# Shuffling can help prevent the model from learning unwanted
# patterns based simply on the order of the training data.
# We normally shuffle the TRAINING data.
# We generally do not need to shuffle the TEST data.
train_loader = DataLoader( train_dataset, batch_size=32, shuffle=True)


# ============================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================

# we are creating a class for our Neural Network architecture.
# In PyTorch, we normally create our own neural network by
# inheriting from nn.Module.
# nn.Module is the base/parent class provided by PyTorch for
# neural network models.
# By inheriting from nn.Module, our BreastCancerNetwork gets
# important functionality from PyTorch such as:
# - parameter management
# - training/evaluation modes
# - saving/loading model parameters
# - compatibility with PyTorch optimizers
# - automatic gradient tracking for its parameters
class BreastCancerNetwork(nn.Module):

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    # __init__ is a special Python method called automatically
    # when we create an object from this class.
    # self refers to the current object/instance of the class.
    # So when we write:
    # self.fc1
    # we are saying:
    # "fc1 belongs to this particular neural network object."
    def __init__(self):

        # super() is used to call the constructor of the parent
        # class.
        # Our parent class is nn.Module.
        # In modern Python, this can also be written simply as:
        # super().__init__()
        # Both forms work here.
        super(BreastCancerNetwork, self).__init__()


        # ----------------------------------------------------
        # FULLY CONNECTED LAYERS
        # ----------------------------------------------------

        # The breast cancer dataset contains 30 input features.
        # Therefore, our first Linear layer receives 30 values
        # for every sample.
        # nn.Linear(30, 64) means:
        # 30 input features
        # 64 neurons
        # Every one of the 64 neurons is connected to all 30
        # input features.
        # The layer also contains trainable weights and biases.
        self.fc1 = nn.Linear(30, 64)

        # The second layer receives the 64 values produced by
        # the first layer and transforms them into 32 values.
        self.fc2 = nn.Linear(64, 32)


        # The final layer receives the 32 values from the previous
        # layer and produces ONE output.
        # We only need one output because this is a binary
        # classification problem.
        # 32 neurons
        #       ↓
        # 1 output neuron
        self.fc3 = nn.Linear(32, 1)


    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    # The forward() method defines how data moves through our
    # neural network.
    # The x parameter represents the input data that is currently
    # being passed through the network.
    def forward(self, x):
        # x is first passed through our first fully connected layer.
        # fc1 performs the mathematical operation:
        # We then apply ReLU to the result.
        # ReLU stands for Rectified Linear Unit.
        # ReLU works approximately like:
        # negative value -> 0
        # positive value -> same value
        # ReLU introduces NON-LINEARITY into our neural network.
        # Without nonlinear activation functions, stacking multiple
        # linear layers would still behave essentially like one
        # linear transformation.
        # The non-linearity allows the neural network to learn
        # more complex relationships in the data.
        x = F.relu(self.fc1(x))

        # The result from the first layer is passed into the
        # second fully connected layer.
        # Again, we apply ReLU so that this layer can learn
        # nonlinear relationships as well.
        x = F.relu(self.fc2(x))

        # The result from the second layer is passed into our
        # final layer.
        # Because this is a binary classification problem, we use
        # sigmoid on the final output.
        # Sigmoid converts the output into a value between 0 and 1.
        # For example:
        # 0.05 -> low probability
        # 0.90 -> high probability
        x = F.sigmoid(self.fc3(x))
        
        # returning the final prediction.
        return x


# ============================================================
# CREATING THE MODEL
# ============================================================

# Now that we have defined our neural network architecture,
# we create an actual object/instance of that architecture.
# This creates our neural network with:
# 30 input features
#       ↓
# 64 neurons
#       ↓
# 32 neurons
#       ↓
# 1 output neuron
# At this point, PyTorch initializes the trainable weights and
# biases of our Linear layers.
model = BreastCancerNetwork()


# ============================================================
# MODEL TRAINING
# ============================================================

# BCELoss stands for Binary Cross Entropy Loss.
# We are using it because our problem is BINARY CLASSIFICATION.
# Our target contains two possible classes, and our final sigmoid
# activation produces a probability between 0 and 1.
# BCELoss compares:
# model prediction
#        vs
# actual target
# and produces a number representing how wrong the prediction is.
# Smaller loss generally means that the predictions are closer
# to the target values.
criterion = nn.BCELoss()

# Adam is our optimizer.
# The optimizer is responsible for updating the trainable weights
# and biases of our neural network after backpropagation.
# model.parameters() gives Adam access to all trainable parameters
# inside our model.
# lr=0.001 is the learning rate.
# The learning rate controls how large the updates to the model's
# weights are during training.
# A very large learning rate can make training unstable.
# A very small learning rate can make training very slow.
optimizer = optim.Adam( model.parameters(), lr=0.001 )


# epochs tells us how many times the model will go through the
# COMPLETE training dataset.
# For example:
# epochs = 20
# means the model will see the complete training dataset 20 times.
epochs = 20

# starting the training loop.
# so the loop runs 20 times.
for epoch in range(epochs):

    # model.train() puts the model into training mode.
    # This is important for certain layers such as Dropout and
    # BatchNorm because their behavior changes between training
    # and evaluation.
    # Our current network does not contain those layers, but it
    # is still good practice to explicitly put the model in
    # training mode.
    model.train()

    # We use this variable to keep track of the total loss
    # accumulated across all batches during the current epoch.
    running_loss = 0.0

    # DataLoader gives us one batch at a time.
    # x_batch contains the input features.
    # y_batch contains the corresponding target values.
    # Since our batch size is 32, x_batch will normally contain
    # 32 samples.
    # Each sample has 30 features.
    # The final batch may contain fewer than 32 samples.
    for x_batch, y_batch in train_loader:
        # Before calculating new gradients, we clear the gradients
        # from the previous batch.
        # PyTorch accumulates gradients by default.
        # If we did not clear them, the gradients from previous
        # batches would be added to the gradients of the current
        # batch.
        optimizer.zero_grad()

        # Forward pass.
        # We send the current batch through our neural network.
        # This internally calls:
        # model.forward(x_batch)
        # The data travels through:
        # 30 -> 64 -> 32 -> 1
        # and we receive the model's predictions.
        preds = model(x_batch)
        
        # calculating the loss.
        # criterion compares:
        # preds   -> what the model predicted
        # y_batch -> what the correct answer actually was
        # The resulting loss tells us how far the predictions
        # are from the actual targets.
        loss = criterion(preds, y_batch)


        # Backpropagation.
        # loss.backward() calculates the gradients of the loss
        # with respect to all trainable parameters in our model.
        # In simple terms, PyTorch works backwards through the
        # operations that produced the loss and determines how
        # each weight contributed to the error.
        loss.backward()

        # Updating the model's parameters.
        # optimizer.step() uses the gradients calculated by
        # loss.backward() to update the weights and biases.
        # This is the actual learning step.
        optimizer.step()

        # loss.item() converts the loss tensor into a normal
        # Python number.
        # We add the current batch's loss to running_loss so that
        # we can see the total loss accumulated during this epoch.
        running_loss += loss.item()

    # printing the loss after the complete epoch.
    # We use epoch + 1 because epoch starts from 0, but humans
    # normally count epochs starting from 1.
    print(f"Epoch {epoch + 1}: Loss was: {running_loss}")


# ============================================================
# MODEL EVALUATION
# ============================================================

# torch.no_grad() tells PyTorch that we are not going to calculate
# gradients inside this block.
# During evaluation, we only want predictions. We are NOT updating
# the model's weights.
# Disabling gradient calculation saves memory and computation.
with torch.no_grad():

    # model.eval() puts the model into evaluation mode.
    model.eval()


    # sending the complete test dataset through the trained model.
    # Notice that we are using x_test_scaled_tensor here instead
    # of x_train_scaled_tensor.
    # The test data was NOT used to update the model's weights.
    # Therefore, it allows us to check how well the model performs
    # on data it did not train on.
    preds = model(x_test_scaled_tensor)


    # calculating the loss on the test dataset.
    # .item() converts the resulting tensor into a normal Python
    # number so that we can print it easily.
    loss = criterion(preds, y_test_tensor ).item()


    # The sigmoid output gives us probabilities between 0 and 1.
    # We use 0.5 as our classification threshold.
    # If:
    # prediction >= 0.5
    #       -> predicted class = 1
    # prediction < 0.5
    #       -> predicted class = 0
    # (preds >= 0.5) creates True/False values.
    # We then compare those predictions with y_test_tensor.
    # The result tells us which predictions were correct.
    # .float() converts:
    # True  -> 1.0
    # False -> 0.0
    # .mean() calculates the percentage of correct predictions.
    # .item() converts the final tensor into a Python number.
    accuracy = (
        ((preds >= 0.5) == y_test_tensor).float().mean().item()
            )

    # printing the final loss and accuracy of our model on the
    # unseen test data.
    print("Test Loss: ", loss)
    print("Model Accuracy: ", accuracy)