from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# REGRESSION ALGORITHMS
# ============================================================

# Regression is a type of supervised machine learning where
# the model learns from input data (x) and known numerical
# target values (y) and then predicts a CONTINUOUS numerical
# value for a new data point.
# We use regression when the output we want is a numerical
# value rather than a category or class.
# Examples:
# House Price              -> Regression
# Temperature              -> Regression
# Salary                   -> Regression
# Sales Amount             -> Regression
# Student Marks            -> Regression
# In the California Housing dataset, the model predicts the
# numerical value of a house based on features describing
# the area where the house is located.
# There are many different regression algorithms.
# Each algorithm learns the relationship between the features
# and the numerical target in a different way.
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
# For regression, common evaluation metrics include:
# - R² (R-squared)
# - Mean Squared Error (MSE)
# - Root Mean Squared Error (RMSE)
# - Mean Absolute Error (MAE)


# ============================================================
# 1. LINEAR REGRESSION
# ============================================================

from sklearn.linear_model import LinearRegression

lr = LinearRegression()


# Linear Regression tries to find a mathematical relationship
# between the input features and the target value.
# The basic idea is to find a line (or hyperplane when there
# are multiple features) that best fits the training data.
# Linear Regression is useful when:
# - The relationship between features and target is
#   approximately linear.
# - We want a simple and fast regression algorithm.
# - We want to understand the effect of individual features.
# - We need a good baseline model.
# Linear Regression is relatively easy to understand and
# interpret.
# However, it may not perform well when the relationship
# between the features and target is highly non-linear.
# IMPORTANT PARAMETERS:
# fit_intercept
# -> Determines whether the model should calculate an
#    intercept/bias term.
# True:
# -> The model calculates the intercept.
# False:
# -> The model assumes the intercept is zero.
# n_jobs
# -> Controls the number of CPU cores used for some
#    computations.
# Linear Regression generally does not require feature scaling,
# although scaling can still be useful when comparing models
# or using it together with other algorithms.


# ============================================================
# 2. RIDGE REGRESSION
# ============================================================

from sklearn.linear_model import Ridge

ridge = Ridge()


# Ridge Regression is basically Linear Regression with
# REGULARIZATION.
# Regularization adds a penalty to large model coefficients.
# This discourages the model from giving extremely large
# weights to features.
# Ridge uses L2 regularization.
# The basic idea is:
# Normal Linear Regression
# -> tries to minimize prediction error.
# Ridge Regression
# -> tries to minimize prediction error + penalty for
#    large coefficients.
# Ridge is useful when:
# - We have many features.
# - Features are highly correlated with each other.
# - Linear Regression is overfitting.
# - We want to keep all features but reduce their influence.
# IMPORTANT HYPERPARAMETER:
# alpha
# alpha controls the strength of regularization.
# Smaller alpha:
# - Weaker regularization.
# - Behaves more like ordinary Linear Regression.
# - Coefficients can become larger.
# Larger alpha:
# - Stronger regularization.
# - Coefficients are pushed closer toward zero.
# - Can reduce overfitting.
# - If too large, the model can underfit.
# Ridge generally benefits from feature scaling because
# regularization operates on the size of the coefficients.


# ============================================================
# 3. LASSO REGRESSION
# ============================================================

from sklearn.linear_model import Lasso

lasso = Lasso()


# Lasso Regression is also Linear Regression with
# regularization.
# However, Lasso uses L1 regularization.
# One important property of Lasso is that it can force some
# feature coefficients to become EXACTLY ZERO.
# This means Lasso can effectively remove less useful features
# from the model.
# For example:
# Feature A -> coefficient = 2.4
# Feature B -> coefficient = 0
# Feature C -> coefficient = -1.7
# Feature B is effectively ignored by the model.
# Lasso is useful when:
# - We have many features.
# - Some features may not be useful.
# - We want automatic feature selection.
# - We want a simpler model.
# IMPORTANT HYPERPARAMETER:
# alpha
# alpha controls the strength of L1 regularization.
# Smaller alpha:
# - Weaker regularization.
# - More features can remain active.
# - Model behaves more like Linear Regression.
# Larger alpha:
# - Stronger regularization.
# - More coefficients can become zero.
# - Can simplify the model.
# - If too large, important features may also be removed.
# Lasso generally benefits from feature scaling because
# regularization is affected by the scale of the features.


# ============================================================
# 4. ELASTIC NET
# ============================================================

from sklearn.linear_model import ElasticNet

elastic = ElasticNet()


# Elastic Net combines BOTH:
# L1 regularization + L2 regularization
# In other words:
# Elastic Net combines the main ideas of Lasso and Ridge.
# L1 regularization:
# -> Can make some coefficients exactly zero.
# L2 regularization:
# -> Shrinks coefficients toward zero.
# Elastic Net is useful when:
# - We have many features.
# - Some features are correlated.
# - We want feature selection.
# - Lasso alone may behave poorly because of correlated
#   features.
# - We want a balance between Lasso and Ridge.
# IMPORTANT HYPERPARAMETERS:
# alpha
# Controls the overall strength of regularization.
# Larger alpha:
# - Stronger regularization.
# Smaller alpha:
# - Weaker regularization.
# l1_ratio
# Controls the balance between L1 and L2 regularization.
# l1_ratio = 1
# -> Pure Lasso-like behavior.
# l1_ratio = 0
# -> Pure Ridge-like behavior.
# Values between 0 and 1:
# -> Combination of L1 and L2 regularization.
# Elastic Net generally benefits from feature scaling.


# ============================================================
# 5. SUPPORT VECTOR REGRESSION (SVR)
# ============================================================

from sklearn.svm import SVR

svr = SVR()

# Support Vector Regression is the regression version of
# Support Vector Machines.
# Instead of trying to classify data into different classes,
# SVR tries to find a function that predicts numerical values.
# The basic idea is to find a function where most training
# points are within a certain acceptable error margin.
# SVR can use KERNELS to model non-linear relationships.
# Common kernels include:
# "linear"
# -> Linear relationship.
# "rbf"
# -> Non-linear relationship.
# "poly"
# -> Polynomial relationship.
# SVR is useful when:
# - The dataset is small or medium-sized.
# - The relationship between features and target is complex.
# - We need a powerful regression algorithm.
# - We want to model non-linear relationships.
# SVR is usually sensitive to feature scales, so scaling the
# features is commonly important.
# IMPORTANT HYPERPARAMETERS:
# C
# Controls the trade-off between:
# - Allowing prediction errors.
# - Making the model fit the training data more closely.
# Smaller C:
# - Allows more errors.
# - Creates a simpler model.
# - Can reduce overfitting.
# Larger C:
# - Penalizes errors more strongly.
# - Tries harder to fit the training data.
# - Can produce a more complex model.
# epsilon
# Defines an acceptable error margin around the prediction.
# Larger epsilon:
# - More errors are tolerated inside the epsilon margin.
# - The model can become less sensitive to small differences.
# Smaller epsilon:
# - The model tries to fit the data more closely.
# kernel
# Determines the mathematical function used to model the
# relationship.
# gamma
# Mainly affects kernels such as "rbf".
# It controls how far the influence of individual training
# examples reaches.
# Smaller gamma:
# - Wider influence.
# - Smoother relationship.
# Larger gamma:
# - More local influence.
# - More complex relationship.
# - Can cause overfitting.


# ============================================================
# 6. DECISION TREE REGRESSOR
# ============================================================

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()


# A Decision Tree Regressor makes predictions by asking a
# sequence of questions about the features.
# Unlike classification trees, which eventually predict a class,
# a regression tree predicts a NUMERICAL VALUE.
# For example:
# Is income > 50000?
#        |
#       YES
#        |
# Is house_age < 10?
#        |
#       YES
#        |
# Predict house value = 4.2
# The tree keeps splitting the data into smaller groups.
# At the final leaf, the model predicts a numerical value based
# on the training samples that reached that leaf.
# Decision Tree Regressors are useful when:
# - The relationship between features and target is complex.
# - The data contains non-linear relationships.
# - We want a model that is relatively easy to understand.
# - We don't want to depend heavily on feature scaling.
# Decision Trees generally do NOT require feature scaling.
# IMPORTANT HYPERPARAMETERS:
# max_depth
# -> Controls the maximum depth of the tree.
# Smaller max_depth:
# - Creates a simpler tree.
# - Can reduce overfitting.
# - May underfit if it is too small.
# Larger max_depth:
# - Allows more complex patterns.
# - Can improve training performance.
# - Can cause overfitting.
# min_samples_split
# -> Minimum number of samples required to split a node.
# Smaller value:
# - Allows more splits.
# - Creates a more complex tree.
# Larger value:
# - Prevents many small splits.
# - Creates a simpler tree.
# min_samples_leaf
# -> Minimum number of samples allowed in a leaf.
# Increasing this value generally makes the tree simpler
# and can help reduce overfitting.
# criterion
# -> Determines how the tree evaluates the quality of a split.
# Common regression criteria include:
# "squared_error"
# "absolute_error"
# "friedman_mse"
# "poisson"


# ============================================================
# 7. RANDOM FOREST REGRESSOR
# ============================================================

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()


# Random Forest Regressor is an ENSEMBLE algorithm.
# Instead of using only one Decision Tree, Random Forest creates
# many different Decision Trees and combines their predictions.
# For regression, the individual trees produce numerical
# predictions and the Random Forest generally combines them
# by averaging their predictions.
# For example:
# Tree 1 -> 4.1
# Tree 2 -> 4.4
# Tree 3 -> 4.2
# Tree 4 -> 4.0
# Random Forest prediction:
# (4.1 + 4.4 + 4.2 + 4.0) / 4
# = 4.175
# Random Forest is useful when:
# - The dataset has complex relationships.
# - The relationship is non-linear.
# - We want a strong general-purpose regression algorithm.
# - We want less overfitting than a single Decision Tree.
# - We don't want to spend too much time manually deciding
#   which features are important.
# Random Forest usually does NOT require feature scaling.
# IMPORTANT HYPERPARAMETERS:
# n_estimators
# -> Determines how many Decision Trees are created.
# Smaller value:
# - Faster training.
# - Fewer trees.
# - Predictions can be less stable.
# Larger value:
# - More trees.
# - Usually more stable predictions.
# - Requires more computation and memory.
# max_depth
# -> Controls the maximum depth of each tree.
# Smaller value:
# - Simpler trees.
# - Can reduce overfitting.
# Larger value:
# - More complex trees.
# - Can learn more detailed patterns.
# - Can increase overfitting.
# max_features
# -> Controls how many features each tree can consider when
#    searching for a split.
# Using different subsets of features makes the trees different
# from each other, which helps the ensemble.
# min_samples_split and min_samples_leaf
# -> Control how easily individual trees create smaller branches
#    and leaves.
# Increasing these values generally makes the trees simpler.


# ============================================================
# 8. K-NEAREST NEIGHBORS REGRESSOR
# ============================================================

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()


# KNN Regression predicts the value of a new data point by
# looking at the closest training data points.
# Unlike KNN Classification, which chooses the most common
# class among the neighbors, KNN Regression usually takes the
# AVERAGE of the neighbors' target values.
# For example, if we use 3 neighbors:
# Neighbor 1 -> 200,000
# Neighbor 2 -> 220,000
# Neighbor 3 -> 210,000
# Prediction:
# (200000 + 220000 + 210000) / 3
# = 210000
# KNN Regression is useful when:
# - Similar data points tend to have similar target values.
# - The dataset is relatively small or medium-sized.
# - The relationship between features and target is not simple.
# - We want a simple algorithm that is easy to understand.
# KNN is sensitive to feature scaling because it uses DISTANCE
# to find the nearest points.
# Therefore, feature scaling is usually very important when
# using KNN Regression.
# IMPORTANT HYPERPARAMETERS:
# n_neighbors
# -> Controls how many neighboring data points are considered.
# Example:
# n_neighbors=3
# -> Looks at the 3 closest points.
# Smaller values:
# - Pays more attention to local patterns.
# - Can learn detailed patterns.
# - Can become sensitive to noise.
# Larger values:
# - Considers a larger area of the dataset.
# - Predictions become smoother.
# - Can reduce the effect of individual noisy points.
# - If too large, important local patterns can be lost.
# weights
# -> Determines how much influence each neighbor has.
# "uniform"
# -> Every neighbor has equal influence.
# "distance"
# -> Closer neighbors have more influence than distant
#    neighbors.
# The value of n_neighbors and other hyperparameters should
# normally be selected based on validation/testing performance.


# ============================================================
# QUICK SUMMARY
# ============================================================

# Linear Regression
# -> Finds a linear mathematical relationship between features
#    and the target.
# -> Fast and easy to understand.
# -> Good baseline for regression.
# -> Works best when the relationship is approximately linear.

# Ridge Regression
# -> Linear Regression + L2 regularization.
# -> Shrinks large coefficients.
# -> Useful when features are correlated.
# -> Helps reduce overfitting.

# Lasso Regression
# -> Linear Regression + L1 regularization.
# -> Can make coefficients exactly zero.
# -> Can perform automatic feature selection.
# -> Useful when some features may be unnecessary.

# Elastic Net
# -> Combines L1 and L2 regularization.
# -> Can perform feature selection while also handling
#    correlated features.
# -> Useful when we want a balance between Lasso and Ridge.

# SVR
# -> Regression version of Support Vector Machine.
# -> Tries to fit predictions within an acceptable error margin.
# -> Can model non-linear relationships using kernels.
# -> Usually benefits from feature scaling.

# Decision Tree Regressor
# -> Makes a sequence of feature-based decisions/splits.
# -> Predicts a numerical value at the final leaf.
# -> Good for non-linear relationships.
# -> Does not normally require feature scaling.
# -> Can easily overfit if allowed to grow too much.

# Random Forest Regressor
# -> Combines many Decision Trees.
# -> Usually averages their numerical predictions.
# -> Good general-purpose algorithm for complex datasets.
# -> Usually does not require feature scaling.
# -> More computationally expensive than one Decision Tree.

# KNN Regressor
# -> Looks at nearby data points.
# -> Predicts using the target values of the nearest neighbors.
# -> Good when similar points have similar target values.
# -> Sensitive to feature scaling.

x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    random_state=42,
                                                    test_size=0.2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)

print("single instance test: ",x_test_scaled[0])
print("single correct value: ",y_test[0])
print("accuracy: ",lr.score(x_test_scaled, y_test))
