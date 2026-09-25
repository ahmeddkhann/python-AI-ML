from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                   random_state=10)

# Parameter Tuning:
# Parameter tuning is used to check different parameters while training
# the model to find out which parameters work the best.
# A model is trained using a specific combination of parameters,
# evaluated, then another combination is tried.
# This process continues until all the combinations specified in
# param_grid have been evaluated.

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}


# cv=3 means that EACH parameter combination will be evaluated
# using 3-fold cross-validation.
# For example, if one parameter combination is:
# n_estimators = 100
# max_depth = 5
# min_samples_split = 2
# the training data is divided into 3 folds.
# The model is trained and tested 3 times:
# Round 1 → Fold 1 is testing, Fold 2 + 3 are training
# Round 2 → Fold 2 is testing, Fold 1 + 3 are training
# Round 3 → Fold 3 is testing, Fold 1 + 2 are training
# The scores from the 3 rounds are then combined to calculate
# the average score for that parameter combination.

clf = RandomForestClassifier(n_jobs=-1)
grid = GridSearchCV(clf, param_grid, cv=3)

# The model searches for the best parameter combination using
# ONLY the training data.
# x_train = input/features
# y_train = target/output
# x_test and y_test must NOT be given to GridSearchCV because
# they should remain completely unseen during parameter tuning.
grid.fit(x_train, y_train)


# best_params_:
# Returns the parameter combination that produced the highest
# cross-validation score during the grid search.
# For example, it might return something like:
# {
#     'max_depth': 10,
#     'min_samples_split': 2,
#     'n_estimators': 200
# }
# This tells us which values of the parameters worked best
# according to the cross-validation results.
print("Best Parameters:", grid.best_params_)


# best_estimator_:
# Returns the actual model that was created using the best
# parameters found by GridSearchCV.
# In this example, it returns a RandomForestClassifier with
# the parameter values stored in best_params_.
# Because refit=True by default, GridSearchCV also refits this
# best model on the entire training dataset after finding
# the best parameter combination.
print("Best Estimator:", grid.best_estimator_)


# best_score_:
# Returns the BEST average cross-validation score obtained
# during the grid search.
# Since RandomForestClassifier uses its default score and we
# have not specified a different scoring metric, this is the
# mean classification accuracy across the 3 validation folds
# for the best parameter combination.
# For example:
# 0.96 means the best parameter combination achieved an
# average cross-validation accuracy of approximately 96%.
# IMPORTANT:
# This score is NOT the final test-set score.
# The separate x_test and y_test should still be used to
# evaluate the final model on completely unseen data.

print("Best Score:", grid.best_score_)
