
# BASIC BREAST CANCER MODEL (BBCM)
# A program to demonstrate how scikit-learn can be used to build
# a simple machine learning classification model using the
# built-in Breast Cancer dataset.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. LOADING THE DATASET

# load_breast_cancer() loads the Breast Cancer dataset provided
# by scikit-learn.
# return_X_y=True tells scikit-learn to return the data directly
# as two separate values:
# x = feature data (input variables)
# y = target data (output variable)
# In machine learning:
# x represents the information that the model uses to make
# a prediction.
# y represents the correct answer that the model is trying
# to predict.
x, y = load_breast_cancer(return_X_y=True)

# ------------------------------------------------------------
# 2. SPLITTING THE DATA INTO TRAINING AND TESTING DATA
# ------------------------------------------------------------

# We divide the dataset into two parts:
# x_train = features used for training the model
# y_train = correct target values used during training
# x_test = features used to test the trained model
# y_test = correct target values used to check the predictions
# test_size=0.2 means that 20% of the complete dataset will
# be used for testing and the remaining 80% will be used
# for training.
# random_state=40 makes the split reproducible.
# This means that every time we run the program, we get
# the same training and testing data.
# If random_state is not specified, the data will normally
# be split differently each time the program runs, which can
# cause slightly different results.
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=40
)


# ------------------------------------------------------------
# 3. SCALING THE FEATURES
# ------------------------------------------------------------

# We use StandardScaler to put the features on a similar scale.
# This is especially important for KNN because KNN is a
# distance-based algorithm.
# KNN calculates the distance between data points to decide
# which training examples are closest to a new data point.
# If one feature has much larger numerical values than another
# feature, it can have a much greater influence on the distance.
# For example:
# Feature A: values between 1 and 10
# Feature B: values between 1,000 and 10,000
# Feature B could dominate the distance calculation simply
# because its numerical values are much larger.
# StandardScaler transforms the features so that they are
# centered around 0 and generally have a standard deviation
# of 1.
scaler = StandardScaler()


# fit_transform() does two things:
# 1. fit()
#    The scaler learns the mean and standard deviation of
#    each feature from the training data.
# 2. transform()
#    It uses those learned values to scale the training data.
# We only fit the scaler on the training data because the
# testing data should remain completely unseen during training.
scaled_x_train = scaler.fit_transform(x_train)


# For the testing data, we only use transform().
# We DO NOT call fit_transform() on x_test.
# The scaler must use the mean and standard deviation that
# it learned from the training data.
# This prevents information from the testing dataset from
# leaking into the training process.
scaled_x_test = scaler.transform(x_test)


# ------------------------------------------------------------
# 4. CREATING THE KNN MODEL
# ------------------------------------------------------------

# KNeighborsClassifier creates a K-Nearest Neighbors (KNN)
# classification model.
# KNN works by looking at the closest training data points
# to a new data point.
# The model then uses the classes of those nearby points
# to decide which class the new data point belongs to.
# By default, KNeighborsClassifier uses 5 neighbors.
knn = KNeighborsClassifier()


# ------------------------------------------------------------
# 5. TRAINING THE MODEL
# ------------------------------------------------------------

# fit() trains the KNN model using:
# scaled_x_train = training features
# y_train        = correct answers for those features
# In the case of KNN, "training" mainly means that the model
# stores the training data so that it can later compare new
# data points with them.
knn.fit(scaled_x_train, y_train)


# ------------------------------------------------------------
# 6. EVALUATING THE MODEL
# ------------------------------------------------------------

# score() evaluates the trained model using the testing data.
# scaled_x_test = testing features
# y_test        = actual/correct target values
# For KNeighborsClassifier, score() returns the accuracy.
# Accuracy means:
#
#     number of correct predictions
#     ------------------------------
#     total number of predictions
# For example, if the model correctly predicts 90 out of
# 100 test samples:
#     accuracy = 90 / 100 = 0.90
# So the result would be 0.90, which represents 90% accuracy.
# IMPORTANT:
# We use the TESTING data here because the purpose is to
# evaluate how well the model performs on data it did not
# use during training.
print(knn.score(scaled_x_test, y_test))

