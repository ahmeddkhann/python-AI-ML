from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
x, y = data.data , data.target
# the train_test_split function is used to split the dataset into training
# and testing sets. the test_size parameter specifies the proportion of
# the dataset to include in the test split. 0.2 means, 20% of the dataset
# will be used for testing, and 80% for training.
# random_state is used to control the shuffling applied to the data before
# applying the split. It ensures that the split is reproducible.
# The reason for this data splitting is to ensure that the model avoids
# overfitting as it is tested by the data that is not present in the
# training dataset. overfitting is the phenomena in which the model
# memorize the features and not learn it.
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2,
                                                    random_state=42)

# Main issue with train_test_split
# Main issue with train_test_split function is that it does not guarantee 
# that the class distribution in the training and testing sets will be the
# same as in the original dataset. This can lead to biased models if the 
# classes are imbalanced. A model might be trained more on one class of a
# data than by other as it will leads to biased model.
# lets visualize it.
import matplotlib.pyplot as plt
import numpy as np

# y_train
# np.bincount() counts how many times each class label appears in y_train.
# In the Iris dataset, the class labels are:
# 0 = setosa
# 1 = versicolor
# 2 = virginica
# For example, if:
# y_train = [0, 1, 0, 2, 1, 0]
# np.bincount(y_train) will return: [3, 2, 1]
# This means:
# class 0 (setosa)      -> 3 samples
# class 1 (versicolor)  -> 2 samples
# class 2 (virginica)   -> 1 sample
train_counts = np.bincount(y_train)

# np.arange(3) creates an array containing [0, 1, 2].
# These numbers represent the positions of the three classes
# on the x-axis of our bar chart.
# Position 0 -> setosa
# Position 1 -> versicolor
# Position 2 -> virginica
train_positions = np.arange(3)

# test_positions/train_positions contains the x-axis positions,
# while train_counts contains the height of each bar.
plt.bar(train_positions, train_counts)
plt.xticks(train_positions, data.target_names)
# plt.show()

test_counts = np.bincount(y_test)
test_positions = np.arange(3)
plt.bar(test_positions, test_counts)
plt.xticks(test_positions, data.target_names)
# plt.show()


#  Now to overcome this issue and splits the data equally class wise
#  we use something like StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

# n_splits=1 means we only want to create ONE train/test split.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

# split(x, y) creates the actual train/test split.
# It returns two arrays of INDEXES:
# train_idx -> indexes of samples selected for training
# test_idx  -> indexes of samples selected for testing
for train_idx, test_idx in split.split(x, y):
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
# ============================================================
# COMMON DATA SPLITTING  METHODS
# ============================================================

# 1. train_test_split
# Randomly splits the dataset into training and testing sets.
# Example:
# 80% -> training
# 20% -> testing
# WHEN TO USE:
# Use it when you need a simple train/test split and don't
# need more advanced cross-validation.

# 2. ShuffleSplit
# Randomly shuffles the dataset and creates train/test splits.
# Unlike train_test_split, it can generate multiple different
# random train/test splits.
# WHEN TO USE:
# Useful when you want to evaluate your model over multiple
# random train/test splits.

# 3. StratifiedShuffleSplit
# Similar to ShuffleSplit, but it tries to preserve the
# class distribution in every split.
# For example, if the original dataset contains:
# Class A -> 70%
# Class B -> 30%
# the generated training and testing sets will try to maintain
# approximately the same 70/30 distribution.
# WHEN TO USE:
# Mainly for classification problems, especially when the
# classes are imbalanced.

# 4. KFold
# Divides the dataset into K separate folds.
# The model is trained on K-1 folds and validated on the
# remaining fold.
# This process is repeated K times so that every fold gets
# used as the validation set once.
# Example with 5 folds:
# Round 1 -> Fold 1 validation, Folds 2-5 training
# Round 2 -> Fold 2 validation, Folds 1,3-5 training
# Round 3 -> Fold 3 validation, Folds 1,2,4,5 training
# Round 4 -> Fold 4 validation, Folds 1,2,3,5 training
# Round 5 -> Fold 5 validation, Folds 1,2,3,4 training
# WHEN TO USE:
# General cross-validation when you want a more reliable
# estimate of model performance than a single train/test split.

# 5. StratifiedKFold
# Similar to KFold, but preserves the class distribution
# across the different folds.
# For example, if the original classification dataset has:
# Class A -> 70%
# Class B -> 30%
# each fold will try to maintain approximately the same
# 70/30 class distribution.
# WHEN TO USE:
# Classification problems where maintaining class proportions
# is important, especially with imbalanced classes.

# 6. RepeatedKFold
# Performs KFold cross-validation multiple times.
# Each repetition creates different folds, allowing the model
# to be evaluated on more combinations of training and
# validation data.
# WHEN TO USE:
# When you want a more robust estimate of model performance
# and have enough computational resources.

# 7. RepeatedStratifiedKFold
# Combines StratifiedKFold with repetition.
# It repeatedly creates folds while trying to preserve the
# class distribution in each fold.
# WHEN TO USE:
# Classification problems where you want repeated
# cross-validation while maintaining class proportions.

# 8. LeaveOneOut (LOO)
# Creates one split for every sample in the dataset.
# In each split:
# - One sample is used for validation/testing.
# - All remaining samples are used for training.
# If the dataset has 100 samples, it creates 100 splits.
# WHEN TO USE:
# Mainly useful for very small datasets.
# IMPORTANT:
# It becomes computationally expensive for large datasets.

# 9. LeavePOut
# Similar to LeaveOneOut, but instead of leaving one sample
# out, it leaves P samples out for validation/testing.
# For example, with P=2:
# - 2 samples -> validation
# - remaining samples -> training
# This is repeated for different combinations of samples.
# WHEN TO USE:
# Mainly for very small datasets where exhaustive validation
# is required.
# IMPORTANT:
# Can become extremely expensive as the dataset gets larger.

# 10. GroupKFold
# Divides data into K folds while keeping samples belonging
# to the same group together.
# A group cannot have some samples in training and other
# samples in validation.
# Example:
# If you have multiple medical records from the same patient,
# all records from that patient stay in the same fold.
# WHEN TO USE:
# When multiple samples belong to the same person, customer,
# patient, device, experiment, etc.

# 11. StratifiedGroupKFold
# Combines the ideas of StratifiedKFold and GroupKFold.
# It tries to:
# - Preserve class distribution.
# - Keep samples belonging to the same group together.
# WHEN TO USE:
# Classification problems where you have both:
# - Groups that must stay together.
# - Class distributions that should be preserved.

# 12. GroupShuffleSplit
# Randomly creates train/test splits based on groups rather
# than individual samples.
# All samples belonging to a particular group are placed
# entirely in either training or testing.
# WHEN TO USE:
# When you need a random train/test split but samples from
# the same group must never appear in both sets.

# 13. TimeSeriesSplit
# Splits data according to chronological order.
# Unlike normal random splitting, it does NOT randomly shuffle
# the data.
# The model always trains on earlier data and validates on
# later data.
# Example:
# Round 1:
# Training -> January
# Testing  -> February
# Round 2:
# Training -> January + February
# Testing  -> March
# Round 3:
# Training -> January + February + March
# Testing  -> April
# WHEN TO USE:
# Time-dependent data such as:
# - Stock prices
# - Weather
# - Sales
# - Website traffic
# - Sensor data
# IMPORTANT:
# Random train/test splitting can cause data leakage in
# time-series problems, so chronological splitting is preferred.




