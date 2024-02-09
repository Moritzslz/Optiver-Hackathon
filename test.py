from sklearn.linear_model import LinearRegression

# Sample data (you should replace this with your actual data)
import numpy as np
n_samples = 100
np.random.seed(0)
YIELDS = np.random.rand(n_samples)
FORD_VOL = np.random.rand(n_samples)
MACRO_VOL = np.random.rand(n_samples)
CURRENCY_VOL = np.random.rand(n_samples)
MUSK_POP = np.random.rand(n_samples)
COMBINATION = (FORD_VOL + MACRO_VOL + CURRENCY_VOL + MUSK_POP) / 4

# Create a Linear Regression model
model = LinearRegression()

# Create the feature matrix and target variable
X = YIELDS.reshape(-1, 1)  # Reshape to make it a 2D array
y = COMBINATION

# Fit the model to the data
model.fit(X, y)

# Get the intercept and coefficient (slope) of the linear regression model
intercept = model.intercept_
slope = model.coef_[0]

# Print the model parameters
print(f'Intercept: {intercept}')
print(f'Slope (Coefficient): {slope}')
