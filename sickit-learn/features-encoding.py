from sklearn.datasets import fetch_openml

# sometimes the data we have is not in the numeric form, but for the 
# machine learning algorithams, the data should must be in numeric form.
# to address this issue, we use feature encoding. we convert the text
# or any form of the data into the numeric form through various techniques
# which is known as features encoding.
# There are several kind of features encoding which are following below.

#1. Ordinal Encoding
# this type of encoding gives the values numeric values in order like
# 0, 1, 2. it is useful when we are dealing with some sort of data that
# is related to each other like sizes which can be small, big or medium
# or speed which can be low, medium and high.
from sklearn.preprocessing import OrdinalEncoder
# fetching a dataset in form of x,y as frame and that will be a dataser
# related to some car data.
x, y = fetch_openml(name="car", as_frame=True, return_X_y=True)
# selecting the columns whom values we want to encode.
columns_to_encode = ["lug_boot", "safety"]
# we have to define the categories that what kind of different categories
# are present in our dataset.
# this code will result in giving following values to the categories.
# column 1. lug_boot
# small -> 0
# med -> 1
# big -> 2
# column 2. safety
# low -> 0
# med -> 1
# high -> 2
encoder = OrdinalEncoder(
    categories=[
        ["small", "med", "big"],
        ["low", "med", "high"]
    ]
)
x[columns_to_encode] = encoder.fit_transform(x[columns_to_encode])
# print(x[columns_to_encode])


# One Hot Encoding
# one hot encoding is used when there are many classifications in a 
# feature of data but they do not have some sort of relationship with
# each other. forexample countries names like pakistan, india, america 
# and china or we can say colors like red, gree, blue or pink. if we
# use ordinal encoding here, it will give advantage to one category over
# the other so avoid this, we use one hot encoding. it keeps the values
# between 0 and 1.
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
# fetching data from the openml and converting it into dataframe
data = fetch_openml(name="adult", as_frame=True).frame
# 1. handle_unkown="ignore":
# imagine a model is trained over some data like in our occupation
# column there is occupations like doctor, engineer, carpenter and
# then after training, in testing or in usage "pilot" appears as an
# occupation so in that case it will throw an error. to prevent error
# we can silently ignore the unkonwn as handle_unkown="ignore".
# 2. sparse_output:
# OneHotEncoder returns many 0 values so to be a more memory effcient
# we choose not to get those 0 matrices by using sparse_output=False.
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# selecting the columns from the dataframe and the transform function
# will creates some new related columns like as we have race it will
# create something like race_white, race_black, race_other and same
# for occupationan as well. 
encoded_values = encoder.fit_transform(data[["occupation", "race"]])
# this line then gives proper names to the newly created columns and this
# returns an array that have names of all the newly created columns. 
new_columns = encoder.get_feature_names_out(["occupation", "race"])
# it creates a new dataframe.. encoded values are the values of each
# columns that we got from the fit_transform() and then we assign the
# columns name to them through columns=new_columns and we tell the
# pandas to use the columns index as data.index and not its own index.
df_encoded = pd.DataFrame(encoded_values, 
                          columns=new_columns,
                          index=data.index)
# at this step, we drop the original columns from the dataset and
# add our encoded coolumns and in this wat, the daatset is converted
# into meaningful numeric values.
final_data = pd.concat(
    [data.drop(columns=["occupation", "race"]), df_encoded], axis=1
)
# print(new_columns)
# print(final_data)