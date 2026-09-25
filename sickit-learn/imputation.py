# For various reasons, many real world datasets contain missing values,
# often encoded as blanks, NaNs or other placeholders. A basic strategy 
# to use incomplete datasets is to discard entire rows and/or columns 
# containing missing values. However, this comes at the price of losing 
# data which may be valuable (even though incomplete). 
# A better strategy is to impute the missing values.
# Now there are two types of imputing values which are following below.

# 1. Univariate/Simple Imputation
from sklearn.impute import SimpleImputer
# simple imputer is used to impute the missing values in a data.
# it looks up to one column at a time and fill the values with two
# different way either by its average value or by median value or 
# by the most repetetive value.
simple_imputer = SimpleImputer(strategy="median")

# 2. Multivariate/Advanved Imputation
# multivariate imputer is used to impute the missing values in a data.
# there are two different types of advanced imputation.
# 1. KNNImputer -> looks upto neighboring columns.
# 2. IterativeImputer -> Iterate over the other columns features

# KNNImputer
from sklearn.impute import KNNImputer
# it looks to n number of columns in a data and fill the values by 
# calculating the values of the other columns as well.
# it will look to two neighboring columns as well in the below example.
knn_imputer = KNNImputer(n_neighbors=2)

# Iterative Imputer
from sklearn.impute import IterativeImputer
# it iteratie over the features of the other columns and then
# suggest an avrage value in that column where the data is missing.
iterative_imputer = IterativeImputer()