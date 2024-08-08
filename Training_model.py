import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


# Load the training data
y_train = pd.read_csv('real_tr.csv', header=None).values.ravel()  
X_train = pd.read_csv(f'tr.csv', header=None)
print(X_train.shape, y_train.shape)
y_test = pd.read_csv('real_te.csv', header=None).values.ravel()  
X_test = pd.read_csv(f'te.csv', header=None)
print(X_test.shape,y_test.shape)

# Define a function to normalize data between 0 and 1
def normalize_segments(X, segment_size = 60):
    X_normalized = X.copy()
    num_rows = X.shape[0]
    for start in range(0, num_rows, segment_size):
        end = min(start + segment_size, num_rows)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_normalized.iloc[start:end] = scaler.fit_transform(X.iloc[start:end])
    return X_normalized


X_train = normalize_segments(X_train)
X_test = normalize_segments(X_test)

num_rows = X_train.shape[0]

# to remove
segment_size = 60
rows_to_remove_start = 10
rows_to_remove_end = 10

indices_to_remove = []

for i in range(0, num_rows, segment_size):
    # Add the first 10 rows of the segment
    for j in range(rows_to_remove_start):
        if i + j < num_rows:
            indices_to_remove.append(i + j)
    
    # Add the last 10 rows of the segment
    for j in range(rows_to_remove_end):
        if i + segment_size - rows_to_remove_end + j < num_rows:
            indices_to_remove.append(i + segment_size - rows_to_remove_end + j)

# Remove the identified rows
X_train = X_train.drop(index=indices_to_remove).reset_index(drop=True)
y_train = np.delete(y_train, indices_to_remove)

param_grid = {'alpha': np.logspace(-3, 3, 7)} 
ridge_reg = Ridge()

# Create a GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(ridge_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
            
# Print the best parameters
print("Best parameters:", grid_search.best_params_)

best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train, y_train)

y_pred = best_rf_model.predict(X_test)
mse = np.mean((y_pred - y_test)**2)
print("Test MSE:", mse)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values', linestyle='dashed', color='blue')
plt.plot(y_pred, label='Predicted Values', linestyle='solid', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('True vs. Predicted Values')
plt.legend()

plt.show()