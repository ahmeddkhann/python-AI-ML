from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# ============================================================
# CLASSIFICATION ALGORITHMS
# ============================================================

# Classification is a type of supervised machine learning where
# the model learns from input data (x) and known class labels (y)
# and then predicts which class a new data point belongs to.
# We use classification when the output we want is a CATEGORY
# or CLASS rather than a continuous numerical value.
# Examples:
# Spam or Not Spam       -> Classification
# Disease or No Disease  -> Classification
# Cat, Dog or Bird       -> Classification
# Pass or Fail           -> Classification
# In this Breast Cancer dataset, the model predicts whether a
# tumor belongs to the malignant or benign class.
# There are many different classification algorithms.
# Each algorithm learns the relationship between the features
# and the target in a different way.
# The general machine learning workflow remains almost the same:
# 1. Load the dataset
# 2. Split the data
# 3. Preprocess the data if necessary
# 4. Create the model
# 5. Train the model using fit()
# 6. Make predictions using predict()
# 7. Evaluate the model
# Usually, the main thing we change is the algorithm/model.
# Different algorithms have different strengths, weaknesses,
# assumptions and hyperparameters.


# ============================================================
# 1. K-NEAREST NEIGHBORS (KNN)
# ============================================================

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()


# KNN classifies a new data point by looking at the closest
# training data points (neighbors).
# For example, if we use 5 neighbors and 4 of them belong to
# class 1 while 1 belongs to class 0, KNN will usually predict
# class 1.
# KNN is useful when:
# - Similar data points usually belong to the same class.
# - The dataset is relatively small or medium-sized.
# - The relationship between features and classes is not simple.
# - We want a simple algorithm that is easy to understand.
# KNN is sensitive to the scale of the features because it uses
# DISTANCE to find the nearest points.
# Therefore, feature scaling is usually very important when
# using KNN.
# IMPORTANT HYPERPARAMETER:
# n_neighbors
# This controls how many neighboring data points KNN looks at.
# Example:
# n_neighbors=3
# -> Looks at the 3 closest points.
# Smaller values:
# - The model pays more attention to very local patterns.
# - It can learn very detailed patterns.
# - It can become sensitive to noise.
# Larger values:
# - The model considers a larger area of the dataset.
# - Predictions become smoother and less sensitive to individual
#   data points.
# - If it becomes too large, important local patterns can be lost.
# The value of n_neighbors should normally be selected based
# on the dataset and validation/testing performance.


# ============================================================
# 2. LOGISTIC REGRESSION
# ============================================================

from sklearn.linear_model import LogisticRegression

lor = LogisticRegression()

# Despite its name, Logistic Regression is a CLASSIFICATION
# algorithm.
# It calculates the probability of a data point belonging to
# a particular class and then uses that probability to make
# the classification.
# Logistic Regression works especially well when the relationship
# between the features and the classes can be separated using
# a relatively simple decision boundary.
# It is useful when:
# - We want a simple and fast classification model.
# - The dataset is relatively large.
# - We want probabilities for predictions.
# - The relationship between features and the target is
#   reasonably linear.
# - We want a model that is relatively easy to interpret.
# Logistic Regression is often a good baseline model because
# it is simple but can perform surprisingly well.
# IMPORTANT HYPERPARAMETERS:
# C
# C controls the strength of regularization.
# Regularization is used to prevent the model from becoming
# unnecessarily complex and overfitting the training data.
# Smaller C:
# - Stronger regularization.
# - The model is more restricted.
# - Can reduce overfitting.
# Larger C:
# - Weaker regularization.
# - The model tries harder to fit the training data.
# - Can potentially increase overfitting.
# max_iter
# This controls the maximum number of iterations allowed for
# the algorithm to find the model parameters.
# A larger value gives the algorithm more time to converge.
# If the model gives a convergence warning, increasing
# max_iter can sometimes solve the problem.
# solver
# This determines the optimization method used to find the
# model parameters.
# Different solvers are useful for different types and sizes
# of datasets.


# ============================================================
# 3. DECISION TREE
# ============================================================

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()


# A Decision Tree makes predictions by asking a sequence of
# questions about the features.
# For example:
# Is feature A > 10?    |
#      YES
#       |
# Is feature B < 5?
#       |
#      YES
#       |
#    Class 1
# The tree keeps splitting the data into smaller groups until
# it reaches a final prediction.
# Decision Trees are useful when:
# - The relationship between features and target is complex.
# - The data contains non-linear relationships.
# - We want a model that is relatively easy to visualize
#   and understand.
# - We don't want to depend heavily on feature scaling.
# One advantage of Decision Trees is that they generally do
# not require feature scaling like KNN does.
# IMPORTANT HYPERPARAMETERS:
# max_depth
# Controls the maximum depth of the tree.
# Smaller max_depth:
# - Creates a simpler tree.
# - Can reduce overfitting.
# - May underfit if it is too small.
# Larger max_depth:
# - Allows the tree to learn more complex patterns.
# - Can improve training accuracy.
# - Can cause overfitting if it becomes too deep.
# min_samples_split
# Determines the minimum number of samples required to split
# an internal node.
# Smaller value:
# - Allows the tree to create more splits.
# - Creates a more complex tree.
# Larger value:
# - Prevents many small splits.
# - Creates a simpler tree.
# min_samples_leaf
# Determines the minimum number of samples that must remain
# in a leaf node.
# Increasing this value generally makes the tree simpler and
# can help reduce overfitting.
# criterion
# Determines how the tree decides which split is better.
# Common options include:
# "gini"
# "entropy"
# "log_loss"
# These methods measure how effectively a split separates
# different classes.


# ============================================================
# 4. SUPPORT VECTOR MACHINE (SVM)
# ============================================================

from sklearn.svm import SVC

svc = SVC()


# Support Vector Machine tries to find a decision boundary that
# separates different classes.
# The main idea is to find a boundary with a large margin between
# the different classes.
# SVM can also use mathematical functions called KERNELS to
# handle data where a simple straight-line boundary is not enough.
# SVM is useful when:
# - The dataset is small or medium-sized.
# - There are many features.
# - The classes have a reasonably clear separation.
# - We need a powerful classification algorithm.
# SVM can work very well on datasets with many features.
# SVM is usually sensitive to feature scales, so scaling the
# features is commonly important.
# IMPORTANT HYPERPARAMETERS:
# C
# Controls the trade-off between:
# - Having a wider/simple decision boundary
# - Correctly classifying training examples
# Smaller C:
# - Allows more classification errors.
# - Produces a softer decision boundary.
# - Can reduce overfitting.
# Larger C:
# - Penalizes classification errors more strongly.
# - Tries harder to correctly classify training examples.
# - Can make the model more complex and potentially overfit.
# kernel
# Determines the type of decision boundary used by SVM.
# Common choices:
# "linear"
# -> Uses a linear decision boundary.
# "rbf"
# -> Can create non-linear decision boundaries.
# "poly"
# -> Uses polynomial relationships.
# "sigmoid"
# -> Uses a sigmoid-based kernel.
# gamma
# Mainly affects kernels such as "rbf".
# It controls how far the influence of an individual training
# example reaches.
# Smaller gamma:
# - Each data point has a wider influence.
# - The decision boundary tends to be smoother.
# Larger gamma:
# - Each data point has a smaller/local influence.
# - The model can create more complex boundaries.
# - Very large values can cause overfitting.


# ============================================================
# 5. RANDOM FOREST
# ============================================================

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()


# Random Forest is an ensemble algorithm.
# Instead of using only one Decision Tree, Random Forest creates
# many different Decision Trees and combines their predictions.
# This usually makes the model more robust than using a single
# Decision Tree.
# Random Forest is useful when:
# - The dataset has complex relationships.
# - We want a strong general-purpose classification algorithm.
# - We have many features.
# - We want less overfitting than a single Decision Tree.
# - We don't want to spend too much time manually deciding
#   which features are important.
# Random Forest usually does not require feature scaling.
# IMPORTANT HYPERPARAMETERS:
# n_estimators
# Determines how many Decision Trees will be created.
# Smaller value:
# - Faster training.
# - Fewer trees.
# - Can produce less stable predictions.
# Larger value:
# - More trees.
# - Usually produces more stable predictions.
# - Takes more computation and memory.
# max_depth
# Controls the maximum depth of each individual tree.
# Smaller value:
# - Simpler trees.
# - Can reduce overfitting.
# Larger value:
# - More complex trees.
# - Can learn more detailed patterns.
# - Can increase overfitting.
# max_features
# Controls how many features each tree is allowed to consider
# when looking for a split.
# Using fewer features makes the individual trees more different
# from each other, which is one of the reasons Random Forest
# works well as an ensemble.
# min_samples_split and min_samples_leaf
# These control how easily the individual trees are allowed
# to create smaller branches and leaves.
# Increasing these values generally makes the trees simpler.


# ============================================================
# 6. NAIVE BAYES
# ============================================================

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()


# Naive Bayes is a probability-based classification algorithm.
# It uses Bayes' theorem to calculate the probability that a
# data point belongs to each class.
# It is called "Naive" because it makes a strong assumption:
# It assumes that the features are conditionally independent
# of each other given the class.
# This assumption is often not completely true in real-world
# datasets, but the algorithm can still work surprisingly well.
# Naive Bayes is useful when:
# - We need a very fast classification algorithm.
# - The dataset is large.
# - We have many features.
# - We are working with text or document classification.
# - We need a simple probability-based model.
# Naive Bayes is commonly used for tasks such as:
# Spam detection
# Text classification
# Sentiment classification
# IMPORTANT HYPERPARAMETER:
# var_smoothing
# GaussianNB assumes that numerical features follow a
# Gaussian (normal) distribution within each class.
# var_smoothing adds a small amount to the feature variance
# to improve numerical stability.
# Smaller value:
# - Uses the calculated variance more directly.
# Larger value:
# - Adds more smoothing.
# - Can make the model less sensitive to very small variances.
# Naive Bayes generally has fewer important hyperparameters
# than algorithms such as Random Forest, SVM or KNN.


# ============================================================
# QUICK SUMMARY
# ============================================================

# KNN
# -> Looks at nearby data points.
# -> Good when similar points belong to similar classes.
# -> Sensitive to feature scaling.

# Logistic Regression
# -> Uses probabilities and a relatively simple decision boundary.
# -> Good baseline classification algorithm.
# -> Fast and relatively easy to interpret.

# Decision Tree
# -> Makes a sequence of decisions/splits.
# -> Good for non-linear relationships.
# -> Does not normally require feature scaling.
# -> Can easily overfit if allowed to grow too much.

# SVM
# -> Finds a decision boundary with a large margin.
# -> Good for small/medium datasets with many features.
# -> Can model non-linear relationships using kernels.
# -> Usually benefits from feature scaling.

# Random Forest
# -> Combines many Decision Trees.
# -> Good general-purpose algorithm for complex datasets.
# -> Usually does not require feature scaling.
# -> More computationally expensive than one Decision Tree.

# Naive Bayes
# -> Uses probabilities and Bayes' theorem.
# -> Very fast and simple.
# -> Particularly useful for text and high-dimensional data.
# -> Makes an independence assumption about features.

# We can train different algorithms on the same dataset and
# compare their performance using appropriate evaluation
# metrics.
#
# Also remember that the algorithm itself is not the only thing
# that affects the model.
#
# Changing the hyperparameters can significantly change how
# the model learns, how complex it becomes, how much it
# overfits or underfits, and how well it performs on unseen data.


x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    random_state=42,
                                                    test_size=0.2)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

clf = KNeighborsClassifier()
clf.fit(x_train_scaled, y_train)
# taking a single instance from the data.
single_instance = x_test_scaled[1]
# clf.predict is used to predict a value of a single instance.
print("Predicted Value: ",clf.predict([single_instance]))
# printing the actual value of the instance.
print("Actual Value: ",y_test[1])
# printing accuracy of the model over the unseen data.
print("Accuracy: ",clf.score(x_test_scaled, y_test))