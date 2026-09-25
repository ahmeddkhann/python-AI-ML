from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

x, y = load_breast_cancer(return_X_y=True)

# Pipelines:
# the pipelines are used to reduce the code and make things easier
# and clearer by putting all the process like scaling, dimensionality
# reducing and algorithams specifier etc inside one pipeline.
# it makes easy to seperate one pipeline from another pipeline.
# all the processes are happens exactly as without pipeline but the
# pipeline keeps one model training seperate from another.

x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=10)
pipe = Pipeline(
    [
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=12)),
        ('foreest', RandomForestClassifier())
    ]
)

pipe.fit(x_train, x_test)