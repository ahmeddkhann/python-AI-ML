# ============================================================
# CLUSTERING ALGORITHMS
# ============================================================

# Clustering is a type of UNSUPERVISED machine learning.
# Unlike classification and regression, we do NOT provide the
# model with known target values (y).
# Instead, we only provide the input data (x), and the algorithm
# tries to discover groups/patterns within the data.
# The groups discovered by a clustering algorithm are called
# CLUSTERS.
# Example:
# Imagine we have customer data:
#
# Age       Income       Spending
# 25        30000        High
# 27        32000        High
# 50        90000        Low
# 52        95000        Low
# We may not tell the model which customers belong together.
# The clustering algorithm can discover groups of customers
# that have similar characteristics.
# Clustering is useful for:
# - Customer segmentation
# - Grouping similar products
# - Image segmentation
# - Document/text grouping
# - Finding unusual patterns
# - Exploratory data analysis
# IMPORTANT:
# Clustering algorithms do not all define a "cluster" in the
# same way.
# Some algorithms look for:
# - Points close to each other
# - Dense regions
# - Similar probability distributions
# - Hierarchical relationships
# - Groups separated by boundaries
# Therefore, the best clustering algorithm depends heavily
# on the shape and structure of the dataset.


# ============================================================
# 1. K-MEANS CLUSTERING
# ============================================================

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x, _ = make_blobs(
    n_samples=5000,
    random_state=10
)

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)

kmean = KMeans(
    n_clusters=5
)

kmean.fit(x_scaled)


# K-Means divides the data into a specified number of clusters.
# The algorithm works roughly like this:
# 1. Choose K cluster centers.
# 2. Assign every data point to the closest center.
# 3. Recalculate the center of each cluster.
# 4. Reassign points to the closest center.
# 5. Repeat until the cluster assignments become stable.
# The center of a cluster is called a CENTROID.

# WHAT TYPE OF DATA IS K-MEANS USEFUL FOR?

# K-Means works best with:
# - Numerical data.
# - Features where distance is meaningful.
# - Compact clusters.
# - Approximately spherical/round clusters.
# - Clusters that are reasonably separated.
# Example:
#       Cluster 1              Cluster 2
#
#          ● ●                    ● ●
#        ● ● ●                  ● ● ●
#          ● ●                    ● ●
# The groups are compact and separated.
# WHY?
# K-Means assigns every point to the nearest centroid.
# Therefore, K-Means works well when the distance between
# a point and a centroid is a good representation of
# similarity.
# For example, imagine customer data:
# Age       Income
# 25        30k
# 27        32k
# 26        31k
# 50        90k
# 52        95k
# 48        88k
# The customers naturally form compact groups based on
# their numerical characteristics.
# K-Means is NOT ideal when:
# - Clusters have complicated shapes.
# - There are many outliers.
# - Clusters have very different densities.
# - Distance to a centroid does not represent similarity.
# For example, with moon-shaped data:
#       ●●●●
#     ●      ●
#    ●        ●
#
# K-Means may divide the data incorrectly because the
# clusters are not spherical.
#
#
# IMPORTANT:
#
# You normally need to tell K-Means how many clusters
# you want.
#
# K-Means is NOT ideal when clusters have complicated shapes.
#
# For example, if the data looks like two crescent/moon shapes,
# K-Means may divide the moons incorrectly because it primarily
# works using distance to cluster centroids.
#
# K-Means is sensitive to feature scale.
#
# Therefore, StandardScaler is commonly used before K-Means.
#
#
# IMPORTANT HYPERPARAMETERS:
#
# n_clusters
# -> Number of clusters to create.
#
# Example:
#
# n_clusters=5
# -> Creates 5 clusters.
#
# Choosing the correct value of K is one of the main challenges
# when using K-Means.
#
#
# init
# -> Determines how the initial cluster centers are selected.
#
# Common option:
#
# "k-means++"
# -> A smarter initialization method that generally gives
#    better starting positions for the centroids.
#
#
# n_init
# -> Determines how many times K-Means is run with different
#    centroid initializations.
#
# More runs can make the result more reliable but require
# more computation.
#
#
# max_iter
# -> Maximum number of iterations for each K-Means run.
#
#
# random_state
# -> Makes the initialization/repeated runs reproducible.
#
#
# DATASET NATURE:
#
# Compact + spherical + numerical
#              ↓
#          K-Means


# ============================================================
# 2. DBSCAN
# ============================================================

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

y, _ = make_moons(
    n_samples=5000,
    random_state=20
)

scaler = StandardScaler()

y_scaled = scaler.fit_transform(y)

dbs = DBSCAN(
    n_jobs=5
)

dbs.fit(y_scaled)


# DBSCAN stands for:
#
# Density-Based Spatial Clustering of Applications with Noise.
#
# Unlike K-Means, DBSCAN does NOT require us to specify the
# number of clusters beforehand.
#
# DBSCAN looks for areas where many data points are close
# together.
#
# These dense areas become clusters.
#
# Sparse areas can be treated as NOISE or OUTLIERS.
#
#
# ============================================================
# WHAT TYPE OF DATA IS DBSCAN USEFUL FOR?
# ============================================================
#
# DBSCAN works best when:
#
# - Data is numerical.
# - Distance between points is meaningful.
# - Clusters form dense regions.
# - Clusters have irregular/non-linear shapes.
# - The dataset contains noise or outliers.
#
# Example:
#
# A cluster might look like:
#
#       ●●●
#     ●     ●
#    ●       ●
#     ●     ●
#       ●●●
#
# This is NOT a spherical cluster.
#
# K-Means would have difficulty representing this structure
# using a single centroid.
#
# DBSCAN does not need a centroid.
#
# Instead, it asks approximately:
#
# "Are there enough points close together?"
#
# Dense region
#      ↓
#   Cluster
#
# Sparse region
#      ↓
# Noise / separation
#
# This makes DBSCAN useful for:
#
# - Geographic/GPS data.
# - Spatial data.
# - Customer locations.
# - Detecting unusual points.
# - Data containing irregular cluster shapes.
#
# Example:
#
# Imagine GPS locations of customers:
#
#       ●●●
#      ●●●●
#
#
#                     ●●
#                    ●●●
#
# DBSCAN can discover these groups without being told
# how many groups exist.
#
#
# DBSCAN is particularly useful for the make_moons dataset
# used above.
#
# The two moon-shaped clusters are not spherical.
#
# DBSCAN can follow the dense curved regions and separate
# the two moons naturally.
#
#
# DBSCAN is NOT ideal when:
#
# - Different clusters have very different densities.
# - One eps value cannot represent all clusters well.
#
# For datasets with strongly different densities,
# OPTICS may be more appropriate.
#
#
# DBSCAN is also sensitive to feature scale, especially
# because it uses distances.
#
# Therefore, scaling is commonly useful.
#
#
# IMPORTANT HYPERPARAMETERS:
#
# eps
# -> Defines the maximum distance for points to be considered
#    neighbors.
#
# Smaller eps:
# - Neighborhoods are smaller.
# - More points may be considered noise.
# - Existing clusters may become fragmented.
#
# Larger eps:
# - Neighborhoods are larger.
# - More points can become connected.
# - Separate clusters may become merged.
#
#
# min_samples
# -> Minimum number of points required for a region to be
#    considered sufficiently dense.
#
# Smaller min_samples:
# - Makes it easier for regions to be considered dense.
# - Can make the algorithm more sensitive to smaller structures.
#
# Larger min_samples:
# - Requires stronger density.
# - More points may be classified as noise.
#
#
# n_jobs
# -> Controls the number of CPU cores used for neighbor
#    searches.
#
#
# IMPORTANT:
#
# DBSCAN labels noise points as:
#
# -1
#
# Other numbers represent cluster IDs.
#
# Example:
#
# -1 -> noise
#  0 -> cluster 0
#  1 -> cluster 1
#  2 -> cluster 2
#
#
# DATASET NATURE:
#
# Dense regions + irregular shapes + noise
#                    ↓
#                  DBSCAN


# ============================================================
# 3. AGGLOMERATIVE CLUSTERING
# ============================================================

from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(
    n_clusters=5
)

agg.fit(x_scaled)


# Agglomerative Clustering is a HIERARCHICAL clustering
# algorithm.
#
# It starts by treating every data point as its own cluster.
#
# Then it repeatedly combines the most similar/closest clusters.
#
# Eventually, many small clusters become fewer and larger
# clusters.
#
# The process can be thought of as:
#
# Individual points
#       ↓
# Small clusters
#       ↓
# Larger clusters
#       ↓
# Final clusters
#
# This creates a hierarchical structure of the data.
#
#
# ============================================================
# WHAT TYPE OF DATA IS AGGLOMERATIVE CLUSTERING USEFUL FOR?
# ============================================================
#
# Agglomerative Clustering is useful when:
#
# - Data is numerical.
# - Distance between observations is meaningful.
# - Groups may contain smaller subgroups.
# - We are interested in hierarchical relationships.
#
# Imagine:
#
#                     ALL DATA
#                         |
#              ---------------------
#              |                   |
#           Group A             Group B
#            /    \              /   \
#          A1      A2           B1    B2
#
# This represents a hierarchy.
#
# The algorithm can help us understand not only the final
# clusters, but also how smaller groups combine into larger
# groups.
#
# WHY?
#
# It starts with:
#
# Point A → Cluster A
# Point B → Cluster B
# Point C → Cluster C
#
# Then:
#
# A + B → larger cluster
#
# Then:
#
# (A+B) + C → even larger cluster
#
# This continues until the desired number of clusters remains.
#
# This makes it useful for data where relationships between
# groups are important.
#
# Example:
#
# Products
#    |
#    +--- Electronics
#    |      |
#    |      +--- Phones
#    |      +--- Laptops
#    |
#    +--- Clothing
#           |
#           +--- Shirts
#           +--- Pants
#
# A hierarchical algorithm can represent these relationships.
#
# Agglomerative Clustering can also be used when clusters
# are not perfectly spherical, depending on the linkage
# method being used.
#
#
# IMPORTANT HYPERPARAMETERS:
#
# n_clusters
# -> Number of clusters to return.
#
#
# linkage
# -> Determines how the distance between two clusters
#    is calculated.
#
# Common options:
#
# "ward"
# -> Tries to minimize the increase in within-cluster
#    variance.
#
# "complete"
# -> Uses the maximum distance between points in two clusters.
#
# "average"
# -> Uses the average distance between points in two clusters.
#
# "single"
# -> Uses the minimum distance between points in two clusters.
#
#
# metric
# -> Determines how distance between points is calculated.
#
# Different linkage methods support different metrics.
#
# Scaling is generally important when distance is being used.
#
#
# DATASET NATURE:
#
# Numerical + meaningful distance + hierarchical relationships
#                         ↓
#                Agglomerative Clustering


# ============================================================
# 4. MEAN SHIFT
# ============================================================

from sklearn.cluster import MeanShift

mean_shift = MeanShift(
    bandwidth=1.0
)

mean_shift.fit(x_scaled)


# Mean Shift is another density-based clustering algorithm.
#
# Instead of explicitly specifying the number of clusters,
# Mean Shift searches for areas where data points are highly
# concentrated.
#
# The algorithm moves points toward areas of higher density.
#
# Eventually, points that move toward the same density peak
# become part of the same cluster.
#
#
# ============================================================
# WHAT TYPE OF DATA IS MEAN SHIFT USEFUL FOR?
# ============================================================
#
# Mean Shift works well when:
#
# - Data is numerical.
# - Distance is meaningful.
# - The dataset contains clear density peaks.
# - We don't know the number of clusters beforehand.
# - Clusters are defined by concentrations of points.
#
# Example:
#
#             ●●●
#           ●●●●●
#             ●●
#
#
#                         ●●
#                       ●●●●
#                        ●●
#
# There are two areas where data points are highly concentrated.
#
# Mean Shift searches for these density peaks.
#
# WHY?
#
# Instead of asking:
#
# "Which centroid is closest?"
#
# Mean Shift asks approximately:
#
# "Where is the highest concentration of points?"
#
# It moves toward regions of higher density.
#
# Points that converge toward the same density peak become
# part of the same cluster.
#
# This means that we don't need to specify:
#
# n_clusters=5
#
# beforehand.
#
# The number of clusters is determined from the density
# structure discovered by the algorithm.
#
#
# Mean Shift is useful for:
#
# - Image segmentation.
# - Pattern discovery.
# - Data with natural density peaks.
# - Problems where the number of groups is unknown.
#
#
# Mean Shift is NOT ideal for very large datasets because
# density estimation and repeated shifting can become
# computationally expensive.
#
#
# IMPORTANT HYPERPARAMETER:
#
# bandwidth
#
# This controls the size of the region used to estimate
# local density.
#
# Smaller bandwidth:
# - Looks at a smaller neighborhood.
# - Can detect smaller/local density peaks.
# - Can produce more clusters.
#
# Larger bandwidth:
# - Looks at a larger neighborhood.
# - Produces smoother density estimates.
# - Can merge smaller density peaks.
#
#
# DATASET NATURE:
#
# Density peaks + unknown number of clusters
#                    ↓
#                Mean Shift


# ============================================================
# 5. GAUSSIAN MIXTURE MODEL (GMM)
# ============================================================

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=5,
    random_state=42
)

gmm.fit(x_scaled)


# Gaussian Mixture Model assumes that the data is generated
# from a mixture of several Gaussian (normal) distributions.
#
# Instead of simply saying:
#
# "This point belongs to cluster 2"
#
# GMM can calculate probabilities such as:
#
# Cluster 0 -> 10%
# Cluster 1 -> 80%
# Cluster 2 -> 10%
#
# This means GMM provides a SOFT clustering approach.
#
# K-Means:
# -> A point belongs to one cluster.
#
# GMM:
# -> A point can have a probability of belonging to
#    different clusters.
#
#
# ============================================================
# WHAT TYPE OF DATA IS GMM USEFUL FOR?
# ============================================================
#
# GMM works well when the data can reasonably be represented
# as a mixture of Gaussian distributions.
#
# The clusters can look approximately like:
#
# - Circles.
# - Ellipses.
# - Oval-shaped distributions.
#
# Unlike K-Means, GMM can model clusters that have different:
#
# - Sizes.
# - Variances.
# - Orientations.
#
# Example:
#
#             ●●●
#           ●●●●●
#         ●●●●
#
#
#                         ●
#                       ●●●
#                     ●●●●●
#
# The two clusters can have different shapes and orientations.
#
# WHY?
#
# GMM models each cluster as a probability distribution.
#
# Therefore, it can represent uncertainty about cluster
# membership.
#
# Imagine a point located between two clusters:
#
# Cluster A → 60%
# Cluster B → 40%
#
# Instead of forcing the point to belong completely to A
# or completely to B, GMM represents the uncertainty.
#
# This is called SOFT CLUSTERING.
#
#
# GMM is useful when:
#
# - Clusters overlap.
# - Cluster membership is uncertain.
# - Data approximately follows Gaussian distributions.
# - Clusters are elliptical rather than perfectly spherical.
#
#
# GMM is NOT ideal when the actual clusters have highly
# irregular shapes such as arbitrary curves or moons.
#
#
# IMPORTANT HYPERPARAMETERS:
#
# n_components
# -> Number of Gaussian distributions/clusters.
#
# Similar to n_clusters in K-Means.
#
#
# covariance_type
# -> Controls the shape and relationship of the covariance
#    matrices.
#
# Common options:
#
# "full"
# -> Each component can have its own full covariance matrix.
# -> Most flexible.
#
# "tied"
# -> All components share the same covariance matrix.
#
# "diag"
# -> Each component has a diagonal covariance matrix.
#
# "spherical"
# -> Each component has a single variance value.
#
#
# max_iter
# -> Maximum number of iterations used during training.
#
#
# random_state
# -> Makes the initialization reproducible.
#
#
# DATASET NATURE:
#
# Gaussian-like + overlapping + elliptical distributions
#                         ↓
#                  Gaussian Mixture


# ============================================================
# 6. SPECTRAL CLUSTERING
# ============================================================

from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(
    n_clusters=5,
    random_state=42
)

spectral.fit(x_scaled)


# Spectral Clustering uses relationships between data points
# to create a graph-like representation of the dataset.
#
# Instead of only looking at the direct geometric distance
# to a centroid like K-Means, it tries to understand how
# points are connected to each other.
#
#
# ============================================================
# WHAT TYPE OF DATA IS SPECTRAL CLUSTERING USEFUL FOR?
# ============================================================
#
# Spectral Clustering is useful when:
#
# - Clusters are non-convex.
# - Clusters have complicated shapes.
# - Local relationships between points are important.
# - The data can naturally be represented as a graph.
# - Simple centroid-based methods don't work well.
#
# Example:
#
# Two moon-shaped clusters:
#
#       ●●●●
#     ●      ●
#    ●        ●
#
#             ●
#              ●
#               ●●●●
#
# These clusters are connected in a complicated/non-linear
# structure.
#
# K-Means tries to divide the data according to distances
# from centroids.
#
# Spectral Clustering instead creates a similarity structure
# between points.
#
# For example:
#
# Point A → strongly connected to B
# Point B → strongly connected to C
# Point C → strongly connected to D
#
# These relationships can help the algorithm identify that
# A, B, C, and D belong to the same structure even when a
# simple centroid-based boundary would not work well.
#
# This is why Spectral Clustering can work well for
# non-convex datasets.
#
#
# Spectral Clustering can be useful for:
#
# - Graph/network data.
# - Social network communities.
# - Image segmentation.
# - Complex spatial structures.
# - Non-linear datasets.
#
#
# Spectral Clustering can become computationally expensive
# on very large datasets because it needs to construct and
# process a similarity/graph structure.
#
#
# IMPORTANT HYPERPARAMETERS:
#
# n_clusters
# -> Number of clusters to create.
#
#
# affinity
# -> Determines how relationships between points are calculated.
#
# Common options:
#
# "rbf"
# -> Uses an RBF kernel to measure similarity.
#
# "nearest_neighbors"
# -> Builds relationships based on nearest neighbors.
#
#
# gamma
# -> Used with certain affinity methods such as RBF.
# -> Controls how quickly similarity decreases with distance.
#
# Smaller gamma:
# - Broader influence.
#
# Larger gamma:
# - More local influence.
#
#
# n_neighbors
# -> Used when affinity="nearest_neighbors".
# -> Determines how many neighbors are used to build
#    the connectivity graph.
#
#
# DATASET NATURE:
#
# Complex relationships + non-convex shapes
#                       ↓
#                Spectral Clustering


# ============================================================
# 7. OPTICS
# ============================================================

from sklearn.cluster import OPTICS

optics = OPTICS(
    min_samples=10
)

optics.fit(y_scaled)


# OPTICS stands for:
#
# Ordering Points To Identify the Clustering Structure.
#
# OPTICS is another density-based clustering algorithm.
#
# It is related to DBSCAN, but it is designed to handle
# datasets where different regions can have different
# densities.
#
#
# ============================================================
# WHAT TYPE OF DATA IS OPTICS USEFUL FOR?
# ============================================================
#
# OPTICS works well when the dataset contains:
#
# - Dense regions.
# - Noise/outliers.
# - Irregular cluster shapes.
# - Clusters with DIFFERENT densities.
#
# Example:
#
# Dense cluster:
#
#       ●●●●
#      ●●●●●
#       ●●●
#
#
# Less dense cluster:
#
#       ●       ●
#          ●
#     ●          ●
#          ●
#
# Both regions may represent real clusters, but their
# densities are very different.
#
# WHY?
#
# DBSCAN generally uses one eps value to define its
# neighborhood.
#
# This can become difficult when:
#
# Cluster A → very dense
# Cluster B → much less dense
#
# One eps value may work well for A but not B.
#
# OPTICS examines the density structure across different
# distance scales.
#
# This makes it more flexible when clusters have different
# density levels.
#
# OPTICS can also identify noise/outliers.
#
#
# Simple comparison:
#
# Similar-density clusters
#          ↓
#        DBSCAN
#
#
# Different-density clusters
#          ↓
#        OPTICS
#
#
# IMPORTANT HYPERPARAMETERS:
#
# min_samples
# -> Controls how many points are required for a
#    sufficiently dense region.
#
# Smaller value:
# - More sensitive to smaller/local structures.
#
# Larger value:
# - Requires stronger density.
# - Can ignore smaller structures.
#
#
# max_eps
# -> Maximum distance considered when searching for neighbors.
#
# Larger max_eps:
# - Allows the algorithm to investigate larger neighborhoods.
#
# Smaller max_eps:
# - Limits the neighborhood size.
#
#
# cluster_method
# -> Determines how clusters are extracted.
#
# Common option:
#
# "xi"
# -> Extracts clusters based on changes in the density
#    structure.
#
# "dbscan"
# -> Allows OPTICS to extract clusters using a DBSCAN-style
#    approach.
#
#
# DATASET NATURE:
#
# Different densities + irregular shapes + noise
#                         ↓
#                       OPTICS


# ============================================================
# QUICK SUMMARY
# ============================================================

# K-Means
# -> Groups points around cluster centroids.
# -> Requires the number of clusters.
# -> Good for compact, roughly spherical clusters.
# -> Sensitive to feature scaling.
#
# Dataset nature:
# -> Numerical + compact + spherical + separated.
#
# Main hyperparameter:
# -> n_clusters


# DBSCAN
# -> Finds dense regions of data.
# -> Can identify noise/outliers.
# -> Does not require the number of clusters beforehand.
# -> Good for irregular/non-linear cluster shapes.
#
# Dataset nature:
# -> Numerical + dense regions + irregular shapes + similar
#    density.
#
# Main hyperparameters:
# -> eps
# -> min_samples


# Agglomerative Clustering
# -> Builds clusters hierarchically by repeatedly merging
#    smaller clusters.
# -> Useful when data has hierarchical relationships.
# -> Usually requires the final number of clusters.
#
# Dataset nature:
# -> Numerical + meaningful distance + hierarchical/group
#    relationships.
#
# Main hyperparameters:
# -> n_clusters
# -> linkage
# -> metric


# Mean Shift
# -> Finds high-density regions / density peaks.
# -> Does not require the number of clusters beforehand.
# -> Useful when clusters are based on density peaks.
#
# Dataset nature:
# -> Numerical + clear density peaks + unknown number
#    of clusters.
#
# Main hyperparameter:
# -> bandwidth


# Gaussian Mixture Model
# -> Assumes data comes from multiple Gaussian distributions.
# -> Provides probability-based/soft cluster assignments.
# -> Can handle overlapping clusters.
# -> Can model elliptical clusters.
#
# Dataset nature:
# -> Numerical + approximately Gaussian + overlapping +
#    elliptical distributions.
#
# Main hyperparameters:
# -> n_components
# -> covariance_type
# -> max_iter


# Spectral Clustering
# -> Uses relationships between points to create clusters.
# -> Useful for complicated/non-convex cluster shapes.
# -> Particularly useful for graph/similarity-based structures.
# -> Requires the number of clusters.
#
# Dataset nature:
# -> Numerical/graph-like + complex relationships +
#    non-convex shapes.
#
# Main hyperparameters:
# -> n_clusters
# -> affinity
# -> gamma
# -> n_neighbors


# OPTICS
# -> Density-based clustering related to DBSCAN.
# -> Can handle different density levels.
# -> Can identify noise.
# -> Does not require the number of clusters beforehand.
#
# Dataset nature:
# -> Numerical + irregular shapes + different densities +
#    noise.
#
# Main hyperparameters:
# -> min_samples
# -> max_eps
# -> cluster_method


# ============================================================
# CLUSTERING ALGORITHM COMPARISON
# ============================================================

# K-Means
# -> Centroid-based
# -> Compact/spherical clusters
# -> Requires number of clusters
# -> Scaling: Usually important
# -> Best when: clusters are compact and well-separated
#
#
# DBSCAN
# -> Density-based
# -> Irregular shapes + noise
# -> Does not require number of clusters
# -> Scaling: Usually important
# -> Best when: clusters are dense and have similar density
#
#
# Agglomerative
# -> Hierarchical
# -> Useful for hierarchical structures
# -> Usually requires number of clusters
# -> Scaling: Usually important
# -> Best when: relationships between groups/subgroups matter
#
#
# Mean Shift
# -> Density-based
# -> Finds density peaks
# -> Does not require number of clusters
# -> Scaling: Usually important
# -> Best when: natural density peaks define the groups
#
#
# Gaussian Mixture
# -> Probability/distribution-based
# -> Gaussian-shaped/overlapping clusters
# -> Requires number of components
# -> Scaling: Usually recommended
# -> Best when: clusters resemble Gaussian distributions
#    and may overlap
#
#
# Spectral Clustering
# -> Graph/similarity-based
# -> Complex/non-convex clusters
# -> Requires number of clusters
# -> Scaling: Usually important
# -> Best when: relationships between points are more
#    important than simple centroid distance
#
#
# OPTICS
# -> Density-based
# -> Different-density clusters + noise
# -> Does not require number of clusters beforehand
# -> Scaling: Usually important
# -> Best when: clusters have different densities
#
#
# ============================================================
# DATASET SHAPE / NATURE CHEAT SHEET
# ============================================================
#
# Compact / spherical
#        ↓
#     K-Means
#
#
# Irregular shape + similar density + noise
#        ↓
#      DBSCAN
#
#
# Hierarchical / group-subgroup relationships
#        ↓
#  Agglomerative
#
#
# Density peaks
#        ↓
#    Mean Shift
#
#
# Gaussian / elliptical / overlapping distributions
#        ↓
#       GMM
#
#
# Complex / non-convex / graph-like relationships
#        ↓
# Spectral Clustering
#
#
# Irregular shape + DIFFERENT densities + noise
#        ↓
#      OPTICS
#
#
# ============================================================
# IMPORTANT FINAL CONCEPT
# ============================================================
#
# The most important thing is NOT to memorize:
#
# "K-Means is algorithm X"
# "DBSCAN is algorithm Y"
#
# Instead, look at the NATURE OF THE DATASET.
#
# Ask:
#
# 1. Are the clusters compact or irregular?
#
# 2. Are the clusters spherical, elliptical, or completely
#    non-linear?
#
# 3. Are the clusters dense?
#
# 4. Do different clusters have different densities?
#
# 5. Are there outliers/noise?
#
# 6. Do I know the number of clusters?
#
# 7. Do I need hard or soft cluster assignments?
#
# 8. Are there hierarchical relationships between groups?
#
# 9. Are relationships between points more important than
#    simple distance to a centroid?
#
#
# Then choose the algorithm whose definition of a
# "CLUSTER" matches the structure of your dataset.
#
# IMPORTANT:
#
# There is no universal "best" clustering algorithm.
#
# Different algorithms can produce completely different
# cluster assignments on the same dataset because each
# algorithm has a different definition of what a cluster is.
#
# Therefore:
#
#                 NATURE OF DATA
#                       ↓
#             Choose the appropriate
#             clustering definition
#                       ↓
#               Clustering Algorithm