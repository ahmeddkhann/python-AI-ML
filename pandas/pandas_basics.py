
# This is a guide to learn python's pandas library thorugh practice
# and coding excercise by using a dataset that consits the DeprecationWarningof 
# 100+ animals 

import pandas as pd
print("="*80)
print("Loading Animals Dataset")
print("="*80)

# printing dataset
# (print first and last 5 rows and columns)
df = pd.read_csv("animals_dataset.csv")
# print(df)
# to print whole dataset
whole_Dataset = df.to_string()
# print(whole_Dataset)
# checking shape of the dataset
# shape includes no. of rows and columns
dataset_shape = df.shape
# print(dataset_shape)
# in our case, 123 rows and 11 coloumns
# checking size of the dataset
# returns the total numbers of values inside a dataset.
dataset_size = df.size
# print(dataset_size)


memory_usage = df.memory_usage(deep=True).sum() / 1024
# print(f"Memory Usage {memory_usage:.2f} kb")
# checking memory usage of the dataset
# memory usage function is used to print the memory usage. it returns the memory
# used by each column. deep=true is used to accurately calculate the memory used
# by strings. sum() adds up the memory usage of each columns. it returns the memory
# usage in bits and bytes so /1024 ensures it will be in kb. .2f displays number 
# in 2 decimal form


# printing start and end rows of the dataset - by default, print 5
# for start rows
head_rows = df.head()
# print(f"head rows: {head_rows}")
# for end rows
end_rows = df.tail()
# print(f"head rows: {end_rows}")
#  for 10 rows from start
no_of_head_rows = df.head(10)
# print(f"head rows: {no_of_head_rows}")
#  for 3 rows from end
no_of_end_rows = df.tail(3)
# print(f"head rows: {no_of_end_rows}")


# Statistics of a dataset
# the describe() function is used to print the statistics of a dataset
basic_statistics = df.describe()
# print(f"Basic Statistics of a dataset: {basic_statistics}")
# It prints the following statistics
# 1. Count -> prints the total number of values in each column.
# 2. Mean -> prints the mean value of the each column.
# 3. Std -> prints the standard deviation of each column
# 4. Min -> prints min value from each column
# 5. Max -> prints max value from each column
# 6. 25% -> returns a value that shows approximately 25% values are beneath it.
# 7. 50% -> returns a value that shows approximately 50% values are beneath it.
# 8. 75% -> returns a value that shows approximately 75% values are beneath it.


# Accessing Individual Columns
name_column = df["Name"]
habitat_column = df["Habitat"]
# print(name_column, habitat_column)
# it by default prints 5 from top and bottom
# to get a specific number of rows from top and bottom
# we will again use head and tail
# n numbers of names from top
n_numbers_of_names = df["Name"].head(4)
# n numbers of habitat from bottom
n_numbers_of_habitat = df["Habitat"].tail(7)
# print(n_numbers_of_names, n_numbers_of_habitat)
# getting column names in a dataset
column_names = df.columns.tolist()
# print(column_names)
# getting data types
datatypes = df.dtypes
# print(datatypes)


# Uniqueness
# print only the unique Names and habiates
unique_name = df["Name"].unique()
unique_habitats = df["Habitat"].unique()
# print the no. of unique values in name and habitats
no_of_unique_names = df["Name"].nunique()
no_of_unique_habitats = df["Habitat"].nunique()
# print(f"unique names: {unique_name}")
# print(f"no of unique names: {no_of_unique_names}")
# print(f"unique habitats: {unique_habitats}")
# print(f"no of unique habitats: {no_of_unique_habitats}")


# Selecting Multiple Columns 
multiple_columns = df[["Name", "Habitat", "Lifespan_years"]]
# print(multiple_columns)
# selecting rows from top of multiple columns
top_rows_from_multiple_columns = multiple_columns.head()
# selecting rows from bottom of multiple columns
bottom_rows_from_multiple_columns = multiple_columns.tail()
# selecting n number of rows from top of multiple columns
n_top_rows_from_multiple_columns = multiple_columns.head(9)
# selecting n number of rows from bottom of multiple column
n_bottom_rows_from_multiple_columns = multiple_columns.tail(3)
# print(f"top 5 rows of multiple columns: {top_rows_from_multiple_columns}")
# print(f"bottom 5 rows from multiple columns: {bottom_rows_from_multiple_columns}")
# print(f"n numbers of top rows of multiple columns: {n_top_rows_from_multiple_columns}")
# print(f"n number of bottom rows from multiple columns: {n_bottom_rows_from_multiple_columns}")


# Selecting Row by its position
# printing first 10 rows
first_10_rows = df.iloc[:10]
# print(first_10_rows)
# printing all the rows after 100th position
rows_after_100_pos = df.iloc[100:]
# print(rows_after_100_pos)
# range of rows from one pos to another 
# printing rows from pos 10 to 30
rows_from_10_to_30 = df.iloc[10:30]
# print(rows_from_10_to_30)


# Specifix Cells
# printing a specific cell by its position
specific_cell = df.iloc[5,2]
# print(specific_cell)
# this will print a specific cell at row 5, column 2
# keep in mind that count starts from 0.


# Boolean Indexing
# Example: 1 - Simple Boolean Indexing
# We will print only those animals whose habitat is Forest.
# Line 1: We access the "Habitat" column and compare each value
# with "Forest". This creates a Boolean Series containing
# True for Forest and False for everything else.
habitat_is_forest = df["Habitat"] == "Forest"
# Line 2: We use the Boolean Series to filter the DataFrame.
# Only the rows where habitat_is_forest is True are selected.
is_forest = df[habitat_is_forest]
# Line 3: We access the "Name" column from the filtered DataFrame.
name_of_forest_animals = is_forest["Name"]
# Print the names of the animals.
# print(name_of_forest_animals)
# Example: 2 - Boolean Indexing with Data Cleaning
# We will print only those animals that are carnivores.
# Line 1: We access the "Is_Carnivore" column and convert
# the different representations of True and False into
# actual Python Boolean values (True or False).
#
# "1.0"    -> True
# "0.0"    -> False
# "True"   -> True
# "False"  -> False
converting_data_into_true_false = df["Is_Carnivore"].replace({
    "1.0": True,
    "0.0": False,
    "True": True,
    "False": False
})
# Line 2: Some rows contain missing values (NaN).
# We replace those missing values with False so that
# the Series can safely be used for Boolean indexing.
converting_data_into_true_false = converting_data_into_true_false.fillna(False)
# Line 3: We use the Boolean Series as a filter.
# Only the rows where Is_Carnivore is True are selected.
is_carnivores = df[converting_data_into_true_false]
# Line 4: From the filtered DataFrame, we select the "Name"
# column and convert it to a string for printing.
is_carnivores = is_carnivores["Name"].to_string()
# Print the names of the carnivorous animals.
# print(is_carnivores)


# Accessing Values from multiple columns
# we will print out the animals that are light weight 
# and have fast speed
# Line1: we are accessing two columns speed and weight and are checking
# if the speed is greater than 50 and weight is less than 100
# also the returned true values are added into a dataframe through df[]
fast_and_light_weight = df[(df["Speed_kmh"] > 50) & (df["Weight_kg"] < 100)]
# Line2: we are accessing that dataframe and are printing the name, weight
# and speed of the returned true values
fast_animals = fast_and_light_weight[["Name", "Weight_kg", "Speed_kmh"]]
# print(fast_animals)


# Acessing values that are either this or that
# we will print the animals name whom contient is either
# asia europe or africa
# Line1: accessing continenet column from dataset and by using isin()
# function, we pass the continent names that will take all the values
# where continent is Asia, Africa or Europe.
is_continent = df["Continent"].isin(["Asia", "Africa", "Europe"])
# Line2: Now we are converting it into a dataframe. we can do it in 
# the above line as well but doing it here so both ways our concepts
# will be cleared
is_asia_or_africa = df[is_continent]
#Line3: Accessing the dataframe and printing name and continets of
# the returned true values.
is_asia_or_africa = is_asia_or_africa[["Name", "Continent"]].to_string()
# print(is_asia_or_africa)


# Adding new columns to the dataset
# we will be adding two different types of columns in which
# one is directly adding a column and one that will be added
# on the basis of the position
# Directly adding a column
# Line1: This first line of code adds a new column named "Light_Weight"
# that check if the weight of animal is less than 50kg or not and on
# the basis of that, it adds a a True or False value
direct_new_column = df["Light_Weight"] = df["Weight_kg"] < 50
# print(df["Light_Weight"])
# adding column on the basis of position
# Line1: This adds a column names is_Fast on position3 which means it 
# will be 4th column and the values will be either True or False by 
# checking if the speed of animal is greater than 50 or not.
position_based_column = df.insert(3, "is_Fast", df["Speed_kmh"] > 50)
# print(df["is_Fast"])


# Droping Columns from the dataset
# this line of code will drop two columns. axis=1 is for
# columns while axis=0 is for rows
drop_colummns = df.drop(["Light_Weight", "is_Fast"], axis=1)
# print(drop_colummns)


# Changing Column Name
# this will change the column names of animal and height
# rename_column = df.rename(columns={
#     "Name": "Animal_Name",
#     "Height_cm": "Height_in_cm"
# })
# print(rename_column.columns.tolist())


# Adding a new row to the dataset
# this code adds a new row at end.
adding_new_row = df.loc[len(df)] = {
        "Animal_ID": 150,
        "Name": "Lion",
        "Habitat": "Forest",
        "Weight_kg": 190,
        "Height_cm": 120,
        "Lifespan_years": 14,
        "Is_Carnivore": True,
        "Endangered": False,
        "Speed_kmh": 80,
        "Population_Status": "Stable",
        "Continent": "Africa"
}
# print(df.tail(3))


# adding/updating a new row on specifix index
# this will add a new row on the index 150
# it will first access the index 150, if it exists, it will update the row
# if it does not exists it will add a new row.
adding_row_on_specific_indx = df.loc[150] = {
        "Animal_ID": 160,
        "Name": "Tiger",
        "Habitat": "Forest",
        "Weight_kg": 110,
        "Height_cm": 220,
        "Lifespan_years": 15,
        "Is_Carnivore": True,
        "Endangered": False,
        "Speed_kmh": 90,
        "Population_Status": "Stable",
        "Continent": "Africa"
}
# print(adding_row_on_specific_indx)


# dropping a row 
# this will drop a row from index 10 means 11th row will be dropped
drop_row = df.drop(index=10)
# print(drop_row.index)


# Using inplace = true; 
# we might see this parameter alot in code which means to change the data
# inside in the original dataframe instead of creating new dataframe.
# in the above examples we have created a new dataframe and then accessed it
# but instead of it if we use inplace=true, so it will directly make chanegs in the
# old dataframe which we do not want at this point.
# example will be:
# rename_column_in_old_df = df.rename(columns={
#     "Name": "Animal_Name",
#      "Height_cm": "Height_in_cm"
# }, inplace=True)
# print(df.columns.tolist())


# GroupBy Operations -> groupby()

# the group by operations are used to make groups of a data and 
# perfrom some kind of operations on it.

# Example 1:
# we will be calculating mean weight of each habitat in the example
# this line of code takes the column "habitat" and each unique value
# becomes a seperate group in our case there are two groups Forest and Savanna
# then ["Weight_kg"] takes the weight of every unique group
# and through .mean() returns mean of it. 
# the sort value is then used to print values from highest to smallest
mean_weight_of_habitats = df.groupby("Habitat")["Weight_kg"].mean().sort_values(ascending=False)
#print(mean_weight_of_habitats)

# Example 2:
# In this example we will also be grouping speed along with weight of habitats
# so groupby("Habitat") will again creates group out of each unique value and then
# we will calculate mean of both weight and speed and then we will sort them to
# print values from highest to lowest. Now the difference is that it will first
# print the value with high mean by weight and if by chance two of weight means are
# exactly same, then the one with greater speed will be printed first then the other.
mean_weight_and_speed_of_habitats = df.groupby("Habitat")[["Weight_kg", "Speed_kmh"]].mean().sort_values(
    by=["Weight_kg", "Speed_kmh"],
    ascending=[False, False]
)
# print(mean_weight_and_speed_of_habitats)

# Example 3:
# we will be calculating multiple values by creating a group using groupby
# we will be calculation mean, max, min, count etc
habitat_aggregate = df.groupby("Habitat").aggregate({
    "Weight_kg": ["mean", "min", "max"],
    "Speed_kmh": ["mean", "min", "max", "std"],
    "Lifespan_years": ["mean", "min", "max", "count"],
    "Height_cm": ["mean", "min", "max",],
})
# print(habitat_aggregate)


# lightest Animals and Fastest Animals
# Lightest Animals
lightest_animals = df.nsmallest(10, "Weight_kg")
print_lightest_animals = lightest_animals[["Name", "Weight_kg"]]
# print(print_lightest_animals)
# Fastest Animals
fastest_animals = df.nlargest(10, "Speed_kmh")[["Name", "Speed_kmh"]]
# print(fastest_animals)


# Corelation Analysis
# corelation is basically finding the relation between different numeric columns
# it shows that how much of a value in a column is effected by change in the another column.
# it outputs range from -1 to 1 where - is negative impact while +  is positive impact
# in example we will check how height, weight, lifespan and speed effects each other.
corelation = df[["Weight_kg", "Speed_kmh", "Height_cm", "Lifespan_years"]].corr()
# print(corelation) 


# Pivot Table
# Pivot table is used to access some data, organize it by rows and columns
# and then summarize it.
# In this example, we will check the mean weight of different habitats
# across the continents
# values = values that will be calculated
# index = the values that will be used as indexes. each unique value will be seperate index
# columns = the values that will be used as columns. each unique value will be seperate column.
# aggfunc = the actual function that should be used for calculations
pivot_table = df.pivot_table(
    values="Weight_kg",
    index="Habitat",
    columns="Continent",
    aggfunc="mean"
)
# print(pivot_table)


# Missing Values
# printing missing values in a column
# isnull() function is used to check if there is some value in the cell or not
# if there is, it return false else return true. then the .sum() function 
# calulcate all the true and return a value. if there are any true, then returned
# value is obviously greater than 0 and that is the condition we are checking in
# the second line and will print only those columns whom values are greater than 0.
missing_values = df.isnull().sum()
total_missing_values = missing_values[missing_values > 0]
# print(total_missing_values)

# Percentage of missing values
# the same story as of above but here the len(df) returns the number of rows in 
# in a dataframe. then we multiply it by 100 to get the perentage and if the
# percentage is greater than 0 we print it.
missing_values_percentage = (df.isnull().sum() / len(df)) * 100
printing_missing_value_percentage = missing_values_percentage[
                                    missing_values_percentage > 0]
# print(printing_missing_value_percentage)

# Droping rows with any missing value
# this will drop all the rows where any value is misiing
# missing_values_dropping = df.dropna()
# print(missing_values_dropping)

# Rows that having missing values
#  this returns all of the rows that have 1 or more missing values in them
rows_with_missing_values = df[df.isnull().any(axis=1)]
# print(rows_with_missing_values[["Name", "Weight_kg"]])

# dropping rows where a value is missing in certain coolumn
# we will drop those rows where weight is null
drop_values_where_Weight_is_missing = df.dropna(
    subset=["Weight_kg"]
)

# we will drop those rows where weight and habitat is null
# the how paramters is used to determine on which basis it should drop row
# it accepts two values, 1. any , 2.all
# if any, if any of weight or habitat is missing the row will be dropped
# if all, only those rows will be dropped where both weight and habitat
# is missing
drop_values_where_Weight_and_habitat_is_missing = df.dropna(
    subset=["Weight_kg", "Habitat"],
    how="all"
)
# print(drop_values_where_Weight_and_habitat_is_missing)


# Filling null values with random values
# fillna() function is used to fill the na values
# Example 1:
# Line1: this line creates a dataframe of name and contient and assigning it to variable
continent_null_values = df[["Continent", "Name"]]
# Line2: then we access the column continent from the variable and fill any
# null value with "Unknown"
continent_null_values["Continent"] = continent_null_values["Continent"].fillna("Unknown")
# print(continent_null_values.to_string())
# Example 2:
# Line1: this line creates a dataframe of name and weight and assigning it to variable
weight_null_values = df[["Weight_kg", "Name"]]
# Line2: then we access the column continent from the variable and fill any
# null value with 100
weight_null_values["Weight_kg"] = weight_null_values["Weight_kg"].fillna(100)
# print(weight_null_values.to_string())


# Handling Duplicate Values
# we will be prinitng all of the duplicate rows in the dataframe
# duplicated() function is used to check the duplicated rows. 
# it checks a row and keep a track of it. then compares it with all of the 
# previous rows and when two similar rows are found, it mark it as duplicated.
# this process runs untill all of the rows are checked.
duplicate_rows = df.duplicated()
duplicate_rows = df[duplicate_rows]
# print(duplicate_rows)
# it also returns the duplicated rows as True and the other as false.
# so we can add .sum() function to calculate total number of duplicated rows.
total_duplicated_rows = df.duplicated().sum()
# print(total_duplicated_rows)


# Keep Operator
# understandig keep operator is essential as it changes behaviour on the basis
# of parameteres on how to deal the duplicate data.
# Mainly used in duplicated() and drop_duplicates() function as keep="" and
# the three possible options are False, First, Last
# 1. False -> All of the duplicated values are marked as True.
# 2. First -> The First time occuring of a value is marked as False 
# while the rest re-occuring are marked True.
# 3. Last -> The last re-occuring of a value is marked False and the rest
# are marked as True.
# main thing to notice is that only those values are calculated as 
# duplicate whom are marked True.
# Use Case:
# First and Last are usually used in drop_duplicates() where we need to keep \
# the value atleast one time and drop the others so it is either first time
# occuring of a value or last time.
# False is used in duplicated() function where we need to see all of the 
# duplicated values. so all of them have specific use cases.


# Drop Duplicate Values
# we will be dropping duplicate values from a specific column
# Example 1:
# this will drop the duplicates values from the column "Animal_ID 
# while keeping the first occurence "
drop_duplicates_except_first = df.drop_duplicates(subset=["Animal_ID"], keep="first")
# print(drop_duplicates_except_first)
# Example 2:
# this will drop the duplicates values from the column "Animal_ID 
# while keeping the last occurence "
drop_duplicates_except_last = df.drop_duplicates(subset=["Animal_ID"], keep="last")
# print(drop_duplicates_except_last)



# Removing white spaces from the strings
# .str.strip() method is used to remove extra spaces from the strings
#  we will be removing extra spaces from the column "Habitat"
remove_extra_spaces = df["Habitat"].str.strip()
# print(remove_extra_spaces)


# Case Standardization
# we will be converting the continent names into lowercase
convert_to_lower_case = df["Continent"].str.lower()
# print(convert_to_lower_case.unique())