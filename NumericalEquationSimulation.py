import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import joblib
import random
from NumericalEquationSimulationFunctions import *

# Number of samples for each wave type
num_samples = 12
num_waves = 100
num_pam = 6000
nodes = 50
k = 1

np.random.seed(42)

# Define the parameter ranges
alpha_values = np.linspace(0.99, 0.5, 10)
beta_values = np.linspace(0.55, 10.5, 10)
phi_values = np.linspace(0, np.pi/4, 5)

# Initialize variables to store the best parameters and the lowest MSE and SER
best_alpha = None
best_beta = None
best_phi = None
lowest_mse = float('inf')
lowest_ser = float('inf')

#wave, wave_repeat, y_true = signal_classification_task(num_samples,num_waves,nodes)
wave, wave_repeat,y_true = PAM4(num_pam, nodes)

# Perform grid search
for alpha in alpha_values:
    print('yes')
    for beta in beta_values:
        for phi in phi_values:

            X = masking(wave_repeat,alpha,beta,phi, k)

            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y_true, test_size=0.2, random_state=42)
            param_grid = {'alpha': np.logspace(-3, 3, 7)}  
            
            # Create a Ridge Regression model
            ridge_reg = Ridge()
            grid_search = GridSearchCV(ridge_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            y_val_pred = best_model.predict(X_val)
            y_val_classify = classify_pam4(y_val_pred)

            # Calculate Mean Squared Error and Symbol Error Rate
            mse_val = mean_squared_error(y_val, y_val_pred)
            ser = np.mean(y_val_classify != y_val)

            # Update the best parameters if the current SER is lower than the lowest SER
            if ser < lowest_ser:
                lowest_mse = mse_val
                lowest_ser = ser
                best_alpha = alpha
                best_beta = beta
                best_phi = phi

                joblib.dump(best_model, 'best_ridge_model.pkl')

print(f"Best alpha: {best_alpha}")
print(f"Best beta: {best_beta}")
print(f"Best phi: {best_phi}")
print(f"Lowest MSE: {lowest_mse}")

# Load the best model for testing with new data
loaded_model = joblib.load('best_ridge_model.pkl')


wave, wave_repeat,y_true = PAM4(3000, nodes)

X = masking(wave_repeat,0.5,0.55,0, k)

y_pred = loaded_model.predict(X)
y_pred_classify = classify_pam4(y_pred)

new_mse = mean_squared_error(y_true,y_pred)
ser = np.mean(y_pred_classify != y_true)

print(f"MSE on new data: {new_mse}")
print(f"SER: {ser}")

X_flat = X.flatten()
plt.figure(figsize=(10, 6))
plt.plot(y_true, label='True Values', linestyle='dashed', color='blue')
plt.plot(y_pred, label='Predicted Values', linestyle='solid', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('True vs. Predicted Values')
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(wave, label='signals')
plt.xlabel('Sample Index')
plt.ylabel('magnitude')
plt.title('Reservoir input signal')
plt.legend()


plt.figure(figsize=(10, 6))
plt.plot(X_flat, label='signals')
plt.xlabel('Reservoir output')
plt.ylabel('magnitude')
plt.title('Reservoir output signal')
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(y_pred_classify, label='signals')
plt.xlabel('Classified')
plt.ylabel('magnitude')
plt.title('Classifed Reservoir output')
plt.legend()

plt.show()

