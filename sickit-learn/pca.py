from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Fetching data from OpenML that contains 28*28 pixels.
# This means there are 784 pixels in one image.
# Each pixel is treated as a separate feature, so every image
# has 784 features.
# For example:
# Image 1 → pixel1, pixel2, pixel3, ..., pixel784
# Image 2 → pixel1, pixel2, pixel3, ..., pixel784
# Therefore:
# Number of features = 28 * 28 = 784

x, y = fetch_openml("mnist_784", return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=40,
    test_size=0.2
)


# Standardize the features before applying PCA.
# PCA is affected by the scale of the features, so scaling
# is commonly performed before PCA.
scaler = StandardScaler()

train_x_scaled = scaler.fit_transform(x_train)
test_x_scaled = scaler.transform(x_test)


# ============================================================
# PCA - PRINCIPAL COMPONENT ANALYSIS
# ============================================================

# PCA is a dimensionality-reduction technique.
# It is used when a dataset contains a large number of features
# and we want to represent the data using fewer dimensions.
# PCA does NOT simply remove individual original features.
# Instead, it creates NEW features called:
#       Principal Components
# These principal components are combinations of the original
# features.
# Example:
# Original dataset:
#       784 features
#            ↓
#           PCA
#            ↓
#       100 principal components
# The model can now work with 100 features instead of 784.

# ============================================================
# HOW DOES PCA DECIDE WHAT TO KEEP?
# ============================================================
# PCA looks at the VARIANCE in the data.
# Variance tells us how much the data varies along a particular
# direction.
# PCA finds directions in the data where the data has the
# greatest amount of variance.
# The first principal component captures the greatest possible
# amount of variance.
# The second principal component captures the next greatest
# amount of variance while being independent (orthogonal) to
# the first component.
# This continues for the remaining components.
# Conceptually:
# Original 784 features
#         ↓
# PCA finds directions with the most variance
#          ↓
# PC1 → most variance
# PC2 → second most variance
# PC3 → third most variance
# ...
# PC784 → least variance
# If we keep only the first few components, we keep the
# directions containing the most variance and discard the
# components containing relatively little variance.
# This allows us to reduce the dimensionality while attempting
# to preserve as much useful information as possible.
# IMPORTANT:
# PCA does NOT know which information is useful for predicting
# the target y.
# PCA is an UNSUPERVISED dimensionality-reduction technique.
# It only looks at X and its variance.
# Therefore, a component with low variance is not necessarily
# useless for the classification problem.
# PCA simply assumes that preserving high-variance directions
# is a useful way to preserve information.


pca = PCA()

# With PCA() and no parameters specified, PCA keeps ALL
# principal components.
# Therefore:
#       784 original features
#               ↓
#       PCA()
#               ↓
#       784 principal components
# There is no actual dimensionality reduction yet.
# To reduce the number of components, we can specify parameters
# such as n_components.


reduced_x_train = pca.fit_transform(train_x_scaled)
reduced_x_test = pca.transform(test_x_scaled)

# ============================================================
# IMPORTANT PCA HYPERPARAMETERS
# ============================================================

# 1. n_components
# This is the most important PCA parameter.
# It controls how many principal components we want to keep.
# Example:
# PCA(n_components=100)
# means:
#       784 features
#            ↓
#       100 principal components
# PCA(n_components=50)
# means:
#
#       784 features
#            ↓
#       50 principal components
#
# Fewer components → greater dimensionality reduction.
# More components → more information/variance retained,
# but less dimensionality reduction.
# n_components can also be a decimal value between 0 and 1.
# Example:
# PCA(n_components=0.95)
# means PCA should automatically select enough components
# to preserve approximately 95% of the variance.
# This is very useful when we don't know beforehand how many
# components we should keep.
# Example:
# pca = PCA(n_components=100)


# 2. whiten
# Default:
# whiten=False
# When whiten=True, PCA scales the principal components so that
# they have unit variance.
# This can sometimes be useful for models that are affected by
# the scale of their input features.
# However, it is not something we normally need to enable
# automatically.
# Example:
# pca = PCA(n_components=100, whiten=True)


# 3. svd_solver
# PCA uses Singular Value Decomposition (SVD) internally to
# calculate the principal components.
# svd_solver controls which method is used to calculate PCA.
# Common options include:
# "auto"
#     Scikit-learn automatically chooses a suitable solver.
# "full"
#     Uses the full SVD calculation.
# "randomized"
#     Uses a randomized method that can be faster for large
#     datasets when we only need a smaller number of components.
# "arpack"
#     Uses a different sparse-matrix method and is useful in
#     certain situations.
# For most normal use cases, leaving it as:
#       svd_solver="auto"
# is fine.


# 4. random_state
# random_state controls randomness when a randomized PCA solver
# is used.
# It allows us to get reproducible results.
# Example:
# PCA(n_components=100, svd_solver="randomized", random_state=42)
# If the randomized solver is not being used, random_state may
# have no effect.


lr = LogisticRegression()
lr.fit(reduced_x_train, y_train)
lr.predict(reduced_x_test)
