# This is a guide to learn numpy library through practice
# and coding exercise with detailed explanations

import numpy as np

# zeros() creates an array filled with 0.0 values
# Parameters: (rows, columns)
zeros_array = np.zeros((2, 3))
# print(f"Zeros Array:\n{zeros_array}")

# ones() creates an array filled with 1.0 values
ones_array = np.ones((2, 4))
# print(f"Ones Array:\n{ones_array}")

# rand() generates random numbers between 0 and 1 from uniform distribution
# Parameters: (rows, columns)
random_array = np.random.rand(3, 4)
# print(f"Random Array:\n{random_array}")

# arange() creates arrays with evenly spaced values
# Parameters: arange(start, stop, step)
# start is inclusive, stop is exclusive, step is increment
range_array = np.arange(1000, 2000, 100)
# print(f"Range Array:\n{range_array}")

# full() creates an array filled with a specific value
# Parameters: full((rows, columns), value)
full_array = np.full((4, 6), 3)
# print(f"Full Array:\n{full_array}")

# linspace() creates arrays with evenly spaced values
# Parameters: linspace(start, stop, num)
# start and stop are both inclusive, num is number of samples
linspace_array = np.linspace(0, 10, 5)
# print(f"Linspace Array:\n{linspace_array}")

# eye() creates an identity matrix (1s on diagonal, 0s elsewhere)
identity_matrix = np.eye(3)
# print(f"Identity Matrix:\n{identity_matrix}")


# Vector: 1-dimensional array
# Contains a single row or column of values
vector = np.array([1, 2, 3, 4, 5])
# print(f"Vector:\n{vector}")
# print(f"Vector shape: {vector.shape}")

# Matrix: 2-dimensional array
# Contains rows and columns of values
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# print(f"Matrix:\n{matrix}")
# print(f"Matrix shape: {matrix.shape}")

# Tensor: multi-dimensional array (3D or more)
# Contains 3 or more dimensions
tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                   [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
# print(f"Tensor:\n{tensor}")
# print(f"Tensor shape: {tensor.shape}")


# Different operations on numpy array
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])

# shape tells us about the dimensions of the array as a tuple (rows, columns)
shape = numpy_array.shape
# print(f"Shape: {shape}")

# size returns the total number of elements in the array
# Calculated as product of all dimensions
size = numpy_array.size
# print(f"Size: {size}")

# dtype tells us about the data type of array elements
# Common types: int32, int64, float32, float64, bool
dtype = numpy_array.dtype
# print(f"Dtype: {dtype}")

# ndim tells us about the number of dimensions
# 1D = vector, 2D = matrix, 3D+ = tensor
dimensions = numpy_array.ndim
# print(f"Dimensions: {dimensions}")

# itemsize returns the size in bytes of each element
item_size = numpy_array.itemsize
# print(f"Item Size: {item_size} bytes")

# nbytes returns the total memory used by the array
# Calculated as: size × itemsize
total_bytes = numpy_array.nbytes
# print(f"Total Bytes: {total_bytes} bytes")


# reshape() changes the shape of an array without changing its data
# New shape must have same total number of elements
# Returns a NEW array (doesn't modify original)
reshaped = numpy_array.reshape(3, 2)
# print(f"Reshaped to (3, 2):\n{reshaped}")

# flatten() converts multi-dimensional array to 1D
# Returns a COPY (doesn't affect original)
flattened = numpy_array.flatten()
# print(f"Flattened: {flattened}")

# ravel() is similar to flatten but returns a VIEW (not a copy)
# More memory efficient if original won't be modified
ravel_array = numpy_array.ravel()
# print(f"Ravel: {ravel_array}")

# transpose() swaps rows and columns
# Accessed with .T or np.transpose()
transposed = numpy_array.T
# print(f"Transposed:\n{transposed}")

# expand_dims() adds a new dimension to the array
# Useful in machine learning where models expect specific input shapes
sample_vector = np.array([1, 2, 3, 4, 5])
# expand at axis=0 (add dimension at beginning)
expand_dims_axis0 = np.expand_dims(sample_vector, axis=0)
# print(f"Expand dims axis=0: {expand_dims_axis0.shape}")
# expand at axis=1 (add dimension at end)
expand_dims_axis1 = np.expand_dims(sample_vector, axis=1)
# print(f"Expand dims axis=1: {expand_dims_axis1.shape}")

# squeeze() removes dimensions of size 1 from the array
# Opposite of expand_dims
array_with_extra_dims = np.array([[[1], [2], [3]]])
squeezed = np.squeeze(array_with_extra_dims)
# print(f"Squeezed shape: {squeezed.shape}")


# Indexing is used to access elements of the array
# 1-dimensional array indexing
sample_1d = np.array([10, 20, 30, 40, 50])
# print(f"Index [0]: {sample_1d[0]}")
# print(f"Index [-1]: {sample_1d[-1]}")  # Last element using negative index

# 2-dimensional array indexing
sample_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(f"Element at [0, 0]: {sample_2d[0, 0]}")
# print(f"Element at [1, 2]: {sample_2d[1, 2]}")

# Slicing is used to access a range of elements
# Format: array[start:end:step]
# start is inclusive, end is exclusive, step is increment
# print(f"[1:4]: {sample_1d[1:4]}")  # Elements from index 1 to 3
# print(f"[::2]: {sample_1d[::2]}")  # Every 2nd element
# print(f"[::-1]: {sample_1d[::-1]}")  # Reverse order

# 2D slicing
# print(f"2D [0:2, 1:3]:\n{sample_2d[0:2, 1:3]}")  # Rows 0-1, Columns 1-2
# print(f"2D [1:, 0]:\n{sample_2d[1:, 0]}")  # All rows from 1 onwards, Column 0

# Boolean indexing is used to filter array elements based on a condition
# Extremely useful for data filtering
condition = sample_1d > 25
# print(f"Boolean array: {condition}")
# print(f"Filtered result: {sample_1d[condition]}")

# Multiple conditions with AND
condition_and = (sample_1d > 15) & (sample_1d < 45)
# print(f"Filtered (AND): {sample_1d[condition_and]}")

# Multiple conditions with OR
condition_or = (sample_1d < 20) | (sample_1d > 40)
# print(f"Filtered (OR): {sample_1d[condition_or]}")

# where() function returns indices where condition is true
# Can also be used for conditional replacement
where_result = np.where(sample_2d > 5)
# print(f"where() row indices: {where_result[0]}")
# print(f"where() column indices: {where_result[1]}")

# where() with replacement: np.where(condition, value_if_true, value_if_false)
replaced = np.where(sample_2d > 5, 0, sample_2d)
# print(f"Replaced elements > 5 with 0:\n{replaced}")


# vstack() stacks arrays vertically (row-wise)
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])
vstacked = np.vstack((array1, array2))
# print(f"VStacked:\n{vstacked}")

# hstack() stacks arrays horizontally (column-wise)
hstacked = np.hstack((array1, array2))
# print(f"HStacked:\n{hstacked}")

# concatenate() is more general stacking function with axis parameter
# axis=0: stack vertically (like vstack)
# axis=1: stack horizontally (like hstack)
concat_axis0 = np.concatenate((array1, array2), axis=0)
# print(f"Concatenate axis=0:\n{concat_axis0}")
concat_axis1 = np.concatenate((array1, array2), axis=1)
# print(f"Concatenate axis=1:\n{concat_axis1}")

# split() splits array into multiple sub-arrays
split_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
split_result = np.split(split_array, 2, axis=1)
# print(f"Split part 1:\n{split_result[0]}")
# print(f"Split part 2:\n{split_result[1]}")


# Mathematical operations apply element-wise to arrays

# sqrt() calculates the square root of each element
math_array = np.array([[1, 4, 9], [16, 25, 36]])
sqrt_result = np.sqrt(math_array)
# print(f"Square root:\n{sqrt_result}")

# log() calculates natural logarithm
log_array = np.array([[1, 2.718, 10], [100, 1000, 10000]])
log_result = np.log(log_array)
# print(f"Natural log:\n{log_result}")

# exp() calculates e^x for each element
exp_array = np.array([[0, 1, 2], [3, 4, 5]])
exp_result = np.exp(exp_array)
# print(f"Exponential:\n{exp_result}")

# sin(), cos(), tan() trigonometric functions
trig_array = np.array([0, np.pi/4, np.pi/2])
sin_result = np.sin(trig_array)
cos_result = np.cos(trig_array)
tan_result = np.tan(trig_array)
# print(f"sin: {sin_result}")
# print(f"cos: {cos_result}")
# print(f"tan: {tan_result}")

# abs() returns absolute value (removes negative sign)
signed_array = np.array([[-3, -1, 2], [4, -5, -6]])
abs_result = np.abs(signed_array)
# print(f"Absolute values:\n{abs_result}")


# Aggregate functions summarize array data

# max() finds the largest value
stats_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
max_val = np.max(stats_array)
# print(f"Max value: {max_val}")
max_per_row = np.max(stats_array, axis=1)
# print(f"Max per row: {max_per_row}")

# min() finds the smallest value
min_val = np.min(stats_array)
# print(f"Min value: {min_val}")

# mean() calculates the average
mean_val = np.mean(stats_array)
# print(f"Mean: {mean_val}")
mean_per_row = np.mean(stats_array, axis=1)
# print(f"Mean per row: {mean_per_row}")

# median() returns middle value when sorted
median_val = np.median(stats_array)
# print(f"Median: {median_val}")

# sum() adds all elements together
sum_val = np.sum(stats_array)
# print(f"Sum: {sum_val}")
sum_per_row = np.sum(stats_array, axis=1)
# print(f"Sum per row: {sum_per_row}")

# std() measures spread of data around the mean
# High std = data spread out, Low std = data clustered
std_val = np.std(stats_array)
# print(f"Standard deviation: {std_val}")

# var() is squared standard deviation
var_val = np.var(stats_array)
# print(f"Variance: {var_val}")


# sort() arranges elements in ascending order
unsorted_2d = np.array([[3, 1, 2], [6, 4, 5]])
sorted_all = np.sort(unsorted_2d.flatten())
# print(f"Sorted (flattened): {sorted_all}")

# sort along axis
# axis=0: sort column-wise
# axis=1: sort row-wise
sorted_axis0 = np.sort(unsorted_2d, axis=0)
# print(f"Sort axis=0:\n{sorted_axis0}")
sorted_axis1 = np.sort(unsorted_2d, axis=1)
# print(f"Sort axis=1:\n{sorted_axis1}")

# argsort() returns indices that would sort the array
simple_array = np.array([3, 1, 4, 1, 5, 9, 2, 6])
sorted_indices = np.argsort(simple_array)
# print(f"Indices that would sort: {sorted_indices}")
# print(f"Sorted using indices: {simple_array[sorted_indices]}")

# unique() finds unique elements in an array
array_with_duplicates = np.array([1, 2, 2, 3, 3, 3, 4])
unique_vals = np.unique(array_with_duplicates)
# print(f"Unique values: {unique_vals}")

# argmax() returns index of maximum value
# argmin() returns index of minimum value
argmax_idx = np.argmax(simple_array)
argmin_idx = np.argmin(simple_array)
# print(f"Index of max: {argmax_idx}, value: {simple_array[argmax_idx]}")
# print(f"Index of min: {argmin_idx}, value: {simple_array[argmin_idx]}")


# Matrix multiplication multiplies two matrices following linear algebra rules
# For matrices A (m×n) and B (n×p), result is (m×p)
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
mat_mult = np.dot(mat1, mat2)
# print(f"Matrix multiplication result:\n{mat_mult}")

# Element-wise multiplication multiplies corresponding elements
element_mult = mat1 * mat2
# print(f"Element-wise multiplication:\n{element_mult}")

# linalg.inv() finds the inverse of a square matrix
# A × A^(-1) = I (identity matrix)
invertible_matrix = np.array([[1, 2], [3, 4]])
# print(f"Original matrix:\n{invertible_matrix}")
try:
    inverse = np.linalg.inv(invertible_matrix)
    # print(f"Inverse:\n{inverse}")
except np.linalg.LinAlgError:
    # print("Matrix is singular (not invertible)")
    pass

# linalg.det() calculates the determinant
# det(A) ≠ 0 means matrix is invertible
determinant = np.linalg.det(invertible_matrix)
# print(f"Determinant: {determinant}")

# linalg.eig() finds eigenvalues and eigenvectors
# Used in: principal component analysis, image compression
symmetric_matrix = np.array([[4, 1], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(symmetric_matrix)
# print(f"Eigenvalues: {eigenvalues}")
# print(f"Eigenvectors:\n{eigenvectors}")

# linalg.norm() calculates the length/magnitude of a vector
vector_norm = np.array([3, 4])
norm = np.linalg.norm(vector_norm)
# print(f"Vector: {vector_norm}")
# print(f"L2 norm (magnitude): {norm}")


# Random numbers generation

# randint() generates random integers within a range
random_ints = np.random.randint(10, 100, size=20)
# print(f"Random integers: {random_ints}")

# rand() generates random floats between 0.0 and 1.0 (uniform distribution)
random_floats = np.random.rand(3, 3)
# print(f"Random floats:\n{random_floats}")

# randn() generates values from normal distribution (bell curve)
# Mean=0, Standard deviation=1
random_normal = np.random.randn(5)
# print(f"Random normal: {random_normal}")

# seed() sets the random seed for reproducibility
# Same seed = same random numbers
np.random.seed(42)
reproducible_1 = np.random.rand(3)
np.random.seed(42)
reproducible_2 = np.random.rand(3)
# print(f"First run: {reproducible_1}")
# print(f"Second run (same seed): {reproducible_2}")


# Broadcasting allows operations on arrays of different shapes
# NumPy automatically expands smaller arrays to match larger ones

# Scalar + Array
scalar = 10
array = np.array([1, 2, 3, 4, 5])
result = array + scalar
# print(f"Array + Scalar: {result}")

# Vector + Matrix (row-wise broadcasting)
vector = np.array([[1], [2], [3]])  # Column vector (3, 1)
matrix = np.array([[10, 20, 30]])   # Row vector (1, 3)
result = vector + matrix
# print(f"Vector + Matrix:\n{result}")

# Practical example with bias (machine learning)
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
bias = np.array([[10, 20, 30]])  # Bias for each column
result = data + bias
# print(f"Data + Bias:\n{result}")


# Copy vs View - Important memory concept

# View (Shallow Copy) - slicing creates a view
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]  # Slicing creates a view
# Modifying the view affects original
# view[0] = 999
# print(f"After modifying view: {original}")  # Original was modified!

# Copy (Deep Copy) - .copy() creates independent array
original2 = np.array([1, 2, 3, 4, 5])
copy = original2[1:4].copy()  # .copy() creates independent array
# Modifying the copy doesn't affect original
# copy[0] = 999
# print(f"After modifying copy: {original2}")  # Original was NOT modified


# Conditional operations and filtering

data_array = np.array([5, 10, 15, 20, 25, 30])

# Single condition
condition = data_array > 15
filtered = data_array[condition]
# print(f"Filtered (> 15): {filtered}")

# Multiple conditions with AND
condition_and = (data_array > 10) & (data_array < 30)
filtered_and = data_array[condition_and]
# print(f"Filtered (AND): {filtered_and}")

# Multiple conditions with OR
condition_or = (data_array < 15) | (data_array > 25)
filtered_or = data_array[condition_or]
# print(f"Filtered (OR): {filtered_or}")

# NOT condition
condition_not = ~(data_array > 20)
filtered_not = data_array[condition_not]
# print(f"Filtered (NOT): {filtered_not}")


# Data type conversion

# astype() converts array to different data type
original_int = np.array([1, 2, 3, 4, 5])
# print(f"Original (int): {original_int}, dtype: {original_int.dtype}")

to_float = original_int.astype(float)
# print(f"As float: {to_float}, dtype: {to_float.dtype}")

to_string = original_int.astype(str)
# print(f"As string: {to_string}, dtype: {to_string.dtype}")

original_bool = np.array([0, 1, 2, 0, 3])
to_bool = original_bool.astype(bool)
# print(f"As bool: {to_bool}")  # 0→False, non-zero→True


# Practical examples

# Data Normalization (Min-Max Scaling)
raw_data = np.array([10, 20, 30, 40, 50])
min_val = np.min(raw_data)
max_val = np.max(raw_data)
normalized = (raw_data - min_val) / (max_val - min_val)
# print(f"Normalized data: {normalized}")  # All values between 0-1

# Standardization (Z-score Normalization)
mean_val = np.mean(raw_data)
std_val = np.std(raw_data)
standardized = (raw_data - mean_val) / std_val
# print(f"Standardized data: {standardized}")  # Mean=0, Std=1

# Finding top N elements
scores = np.array([85, 92, 78, 95, 88, 82, 90, 79])
n = 3
top_indices = np.argsort(scores)[-n:][::-1]
# print(f"Top {n} indices: {top_indices}")
# print(f"Top {n} scores: {scores[top_indices]}")

# Batch processing
all_data = np.arange(12).reshape(4, 3)  # 4 samples, 3 features
batch_size = 2
# for i in range(0, len(all_data), batch_size):
#     batch = all_data[i:i+batch_size]
#     print(f"Batch {i//batch_size + 1}:\n{batch}")