from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# cross validation is a model evaluation technique which splits
# the data into n number of batches. the model is trained over n-1 batches
# the remaining batch is used to test the model accuracy. this process is
# repeated unless all of the batches are not tested.
x,y = load_breast_cancer(return_X_y=True)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
clf = KNeighborsClassifier()
# in this line, cv=5 means that data will be divided into 5 different
# batches. batch1, batch2, batch3, batch4 and batch5. 
# model trained on batch 5,2,3,4 -> batch 1 will be used for testing.
# model trained on batch 1,5,3,4 -> batch 2 will be used for testing.
# model trained on batch 1,2,5,4 -> batch 3 will be used for testing.
# model trained on batch 1,2,3,5 -> batch 4 will be used for testing.
# model trained on batch 1,2,3,4 -> batch 5 will be used for testing.

# scoring: It tells cross-validation what metric should be used to 
# evaluate the model.
# possible options: there are multiple possible options for the 
# classification and regression models.
# for classification:
# accuracy, precision, f1, recall, roc_auc
# for regression:
# r2, neg_mean_squarred_error
# n_jobs: This controls how many CPU cores/jobs are used to
# perform cross-validation. -1 uses all cpu cores.
cross_validation = cross_val_score(clf, x_scaled, y,
                                   cv=5, scoring="accuracy", n_jobs=2)
print("Cross Validation: ",cross_validation)