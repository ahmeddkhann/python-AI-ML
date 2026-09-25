from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Pre-processing is used to pre-process the data in some way before it is
#  used to train and test the model. there are serveral algorithams 
# that can be used to pre-process the data which are following bellow.

# 1. StandardScaler
from sklearn.preprocessing import StandardScaler
# standard scaler is used when there are different types of values having
# very different ranges like age which can be from 0 to 100 and salary 
# which can be from 10000 to 1000000 so if we compare both, they have very
# huge difference so model might be biased towards one values so for that
# we use standard scaler.
scaler = StandardScaler()
# there are two diff types of methods in which a model can learn the
# features and then transform it into the scaler data. 
# 1. we can eithet use fit() so the model will learn the features and then
# transform() so that the data will be transformed. 
# 2. we either use fit_transform() so the model learns features and scale
# data at once. we usually use it for training. for testing, as the 
# features are already learned, we use only transform() on it, so that
# the data is scaled.
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

# 2. MinMax Scaler
from sklearn.preprocessing import minmax_scale
# min max scaler is used when we want to scale the features and get a range
# of it between 0 and 1. so what it does is that scale the features
# into values between 0 and 1.
scaler = minmax_scale()

# RobustScaler
from sklearn.preprocessing import RobustScaler
# robust scaler is used when there is many outliers in the data. it is
# used t scale the data in a way that the outliers does not affect the 
# training and testing of the model.
scale_outliers = RobustScaler()

# MaxAbsScaler
from sklearn.preprocessing import maxabs_scale
# maxabs_scale is used to transform the data into the range of -1 to 1.
# it is used on sparse data (data that have many zero values). the 0
# remains 0 while the rest of the data is transformed into -1 to 1.
