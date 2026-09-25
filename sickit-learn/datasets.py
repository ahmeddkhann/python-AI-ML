# This file is about different types of datasets in scikit-learn. 
# It provides functions to load and fetch various datasets, 
# including toy datasets, real-world datasets, and synthetic datasets. 
# The datasets can be used for machine learning tasks such as classification, 
# regression, and clustering. It also provides making custom datasets for testing 
# and benchmarking machine learning algorithms.



# There are primarly 3 differrent types of datasets in scikit-learn:

# 1. Loading datasets: These are small datasets that are included 
# with scikit-learn and can be loaded using functions such as load_iris(), 
# load_digits(), and load_boston(). These datasets are useful for
# testing and benchmarking machine learning algorithms.
# different loading datasets and there use cases:
# 1. load_iris(): This function loads the iris dataset, 
# which is a classic dataset for classification tasks.
# 2. load_digits(): This function loads the digits dataset,
# which is a dataset of handwritten digits for classification tasks.
# 3. load_boston(): This function loads the Boston housing dataset,
# which is a dataset for regression tasks. 
# 4. load_breast_cancer(): This function loads the breast cancer dataset,
# which is a dataset for binary classification tasks.
# 5. load_wine(): This function loads the wine dataset,
# which is a dataset for multi-class classification tasks.
# 6. load_diabetes(): This function loads the diabetes dataset,
# which is a dataset for regression tasks.
# 7. load_linnerud(): This function loads the linnerud dataset,
# which is a dataset for multi-output regression tasks.
# 8. load_sample_image(): This function loads a sample image dataset,
# which is a dataset for image processing tasks.
# 9. load_sample_images(): This function loads a sample images dataset,
# which is a dataset for image processing tasks.
# 10. load_sample_spectra(): This function loads a sample spectra dataset,
# which is a dataset for spectral analysis tasks.
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import LogisticRegression
from sklearn.metrics import accuracy_score

loadX, loadY = load_wine(return_X_y=True)
# Splitting the dataset into training and testing sets.
load_x_train, load_x_test, load_y_train, load_y_test = train_test_split(
                                                        loadX, loadY,
                                                        test_size=0.2,
                                                        random_state=42
)

# Scaling the data to ensure that all features are on the same scale.
load_scaler = StandardScaler()
load_scaled_x_train = load_scaler.fit_transform(load_x_train)
load_scaled_x_test = load_scaler.transform(load_x_test)

# Training the model using Logistic Regression algorithm and testing it.
# Logistic Regression is a linear model that can not give solution
# directly for multi-class classification tasks. It needs iterations
# again and again to find the best solution. through the max_iter,
# we can set the maximum number of iterations to find the best solution.
# if it finds before 1000, the model will stop and give the solution. 
# and won't iteration again and again. if it doesn't find the best solution in 1000 iterations,
# it will stop and give the best solution it found in 1000 iterations.
load_model = LogisticRegression(max_iter=1000)
load_model.fit(load_scaled_x_train, load_y_train)

# Predicting the target variable using the testing data.
load_y_pred = load_model.predict(load_scaled_x_test)

# Evaluating the model using accuracy score.
load_accuracy = accuracy_score(load_y_test, load_y_pred)
print("Accuracy of the model is: ", load_accuracy)

\


# 2. Fetching datasets: These are larger datasets that are not included
# with scikit-learn but can be downloaded from the internet using functions
# such as fetch_20newsgroups(), fetch_lfw_people(), and fetch_covtype(). 
# These datasets are useful for real-world machine learning tasks.
# different fetching datasets and there use cases:
# 1. fetch_20newsgroups(): This function fetches the 20 newsgroups dataset,
# which is a dataset for text classification tasks.
# 2. fetch_lfw_people(): This function fetches the Labeled Faces in the Wild 
# (LFW) dataset, which is a dataset for face recognition tasks.
# 3. fetch_covtype(): This function fetches the Covertype dataset,
# which is a dataset for multi-class classification tasks.
# 4. fetch_openml(): This function fetches datasets from the OpenML repository, 
# which is a collection of datasets for various machine learning tasks.
# 5. fetch_mldata(): This function fetches datasets from the mldata.org 
# repository, which is a collection of datasets for various machine learning tasks.
# 6. fetch_california_housing(): This function fetches the California housing 
# dataset, which is a dataset for regression tasks.
# 7. fetch_kddcup99(): This function fetches the KDD Cup 1999 dataset, 
# which is a dataset for network intrusion detection tasks.
# 8. fetch_rcv1(): This function fetches the Reuters Corpus Volume I (RCV1) 
# dataset, which is a dataset for text classification tasks.
# 9. fetch_olivetti_faces(): This function fetches the Olivetti faces dataset,
# which is a dataset for face recognition tasks.
# 10. fetch_mnist(): This function fetches the MNIST dataset, which is a 
# dataset of handwritten digits for classification tasks.
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

fetchX, fetchY = fetch_california_housing(return_X_y=True)
# Splitting the dataset into training and testing sets.
fetch_x_train, fetch_x_test, fetch_y_train, fetch_y_test = train_test_split(
                                                        fetchX, fetchY,
                                                        test_size=0.2, 
                                                        random_state=42)

# Scaling the data to ensure that all features are on the same scale.
fetch_scaler = StandardScaler()
fetch_scaled_x_train = fetch_scaler.fit_transform(fetch_x_train)
fetch_scaled_x_test = fetch_scaler.transform(fetch_x_test)

# Training the model using Linear Regression algorithm and testing it.
fetch_model = LinearRegression()
fetch_model.fit(fetch_scaled_x_train, fetch_y_train)

# Predicting the target variable using the testing data.
fetch_y_pred = fetch_model.predict(fetch_scaled_x_test)

# Evaluating the model using mean squared error and R-squared score.
# mean squared error is a measure of how well the model fits the data.
# it compares the actual value and the predicted value and returns
# the average of the squared differences.
fetch_mse = mean_squared_error(fetch_y_test, fetch_y_pred)

# R-squared score is a measure of how well the model explains the 
# variance/change in the data.
fetch_r2 = r2_score(fetch_y_test, fetch_y_pred)
print("Mean Squared Error of the model is: ", fetch_mse)
print("R-squared score of the model is: ", fetch_r2)




# 3. Making datasets: These are synthetic datasets that can be generated
# using functions such as make_classification(), make_regression(), and
# make_blobs(). These datasets are useful for testing and benchmarking
# machine learning algorithms.
# different making datasets and there use cases:
# 1. make_classification(): This function generates a synthetic dataset
# for classification tasks. 
# 2. make_regression(): This function generates a synthetic dataset for
# regression tasks. 
# 3. make_blobs(): This function generates a synthetic dataset for clustering
# tasks. 
# 4. make_moons(): This function generates a synthetic dataset for binary
# classification tasks. It creates two interleaving half circles.
# 5. make_circles(): This function generates a synthetic dataset for binary
# classification tasks. It creates a large circle containing a smaller circle.
# 6. make_swiss_roll(): This function generates a synthetic dataset for
# regression tasks. It creates a 3D swiss roll shape.
# 7. make_s_curve(): This function generates a synthetic dataset for regression
# tasks. It creates a 3D S-shaped curve.
# 8. make_checkerboard(): This function generates a synthetic dataset for
# classification tasks. It creates a checkerboard pattern.
# 9. make_gaussian_quantiles(): This function generates a synthetic dataset
# for classification tasks. It creates Gaussian quantiles.
# 10. make_low_rank_matrix(): This function generates a synthetic dataset for
# regression tasks. It creates a low-rank matrix with specified rank and noise level.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Generating a dataset.
# n_samples is the number of samples to generate.
# n_features is the number of features to generate.
# n_informative is the number of informative features to generate.
# n_redundant is the number of features that are generated
# from the combination of the informative features.
# random_state is used to ensure that the same dataset is generated
make_x , make_y = make_classification(n_samples=1000, 
                            n_features=20,
                            n_informative=2, 
                            n_redundant=10, 
                            random_state=42)
# Splitting the dataset into training and testing sets.
make_x_train, make_x_test, make_y_train, make_y_test = train_test_split(make_x, make_y,
                                                    test_size=0.2, 
                                                    random_state=42)

# Scaling the data to ensure that all features are on the same scale.
make_scaler = StandardScaler()
make_scaled_x_train = make_scaler.fit_transform(make_x_train)
make_scaled_x_test = make_scaler.transform(make_x_test)

# Training the model using Logistic Regression algorithm and testing it.
make_model= LogisticRegression()
make_model.fit(make_scaled_x_train, make_y_train)

# Predicting the target variable using the testing data.
make_y_pred = make_model.predict(make_scaled_x_test)

# Evaluating the model using accuracy score.
accuracy = accuracy_score(make_y_test, make_y_pred)
print("Accuracy of the model is: ", accuracy)