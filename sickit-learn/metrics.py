
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Metrics are used to evaluate how good a model is.
# There are different types of metrics for classification
# and regression algorithms.

# ============================================================
# CLASSIFICATION METRICS
# ============================================================

from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    random_state=42,
                                                    test_size=0.2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

knn = KNeighborsClassifier()
knn.fit(x_train_scaled, y_train)
y_pred = knn.predict(x_test_scaled)


# Precision:
# Precision tells us, out of all the samples that the model
# predicted as positive, how many were actually positive.
# High precision means fewer False Positive predictions.
# Formula:
# Precision = True Positives / (True Positives + False Positives)
print("Precision: ", precision_score(y_test, y_pred))

# Accuracy:
# Accuracy tells us how many predictions the model got correct
# out of all the predictions it made.
# It works well when the classes are reasonably balanced.
# Formula:
# Accuracy = Correct Predictions / Total Predictions
print("Accuracy: ", accuracy_score(y_test, y_pred))


# Recall:
# Recall tells us, out of all the samples that were actually
# positive, how many the model successfully identified.
# High recall means fewer False Negative predictions.
# Formula:
# Recall = True Positives / (True Positives + False Negatives)
print("Recall: ", recall_score(y_test, y_pred))


# F1 Score:
# F1 score combines Precision and Recall into a single metric.
# It is useful when we want a balance between precision and recall.
# F1 score becomes high when both precision and recall are high.
# Formula:
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
print("F1 Score: ", f1_score(y_test, y_pred))


# ============================================================
# REGRESSION METRICS
# ============================================================

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor

x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    random_state=42,
                                                    test_size=0.2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

knr = KNeighborsRegressor()
knr.fit(x_train_scaled, y_train)
y_pred = knr.predict(x_test_scaled)


# R² Score:
# R² measures how well the model explains the variation in
# the target values.
# R² = 1 means perfect predictions.
# R² = 0 means the model performs approximately like predicting
# the mean of the target values.
# R² can also be negative when the model performs worse than
# that simple mean-based prediction.
# HIGHER R² is generally better.
print("R² Score: ", r2_score(y_test, y_pred))


# Mean Absolute Error (MAE):
# MAE calculates the average absolute difference between the
# actual values and predicted values.
# It tells us, on average, how far our predictions are from
# the actual values.
# MAE is easy to understand because it uses the same unit
# as the target value.
# LOWER MAE is better.

print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))


# Mean Squared Error (MSE):
# MSE calculates the average of the squared differences between
# actual and predicted values.
# Because the errors are squared, large errors are punished
# much more heavily than small errors.
# LOWER MSE is better.
# MSE is expressed in squared units of the target.

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))


# Root Mean Squared Error (RMSE):
# RMSE is simply the square root of MSE.
# Like MSE, it gives more importance to large errors.
# However, because we take the square root, RMSE is expressed
# in the same unit as the original target.
# LOWER RMSE is better.

print("Root Mean Squared Error: ", root_mean_squared_error(y_test, y_pred))
