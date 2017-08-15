import pandas as pd

# Read data.csv and treat the 0th row as a header containing column names
x = pd.read_csv("data.csv", header=0)
x

# See the shape of the data
x.shape

# Get the first two rows of x as a DataFrame
x.head(2)

# Get the rows with a value of less than 0.5 in the 'X0' column.
x[x['X0'] < 0.5]

# Get the values from the 0th row as a Series
x.iloc[0]

# Get the values from the 0th column as a Series
x['X0']

# Get a representation of the series as a numpy matrix
x.as_matrix()
