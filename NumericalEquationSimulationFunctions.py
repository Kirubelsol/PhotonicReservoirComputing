import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import joblib
import random

def signal_classification_task(num_samples,num_waves,nodes):

    # Generate sine wave values (adjusted by -0.5)
    time_points = np.linspace(0, np.pi, num_samples)
    sine_values = np.sin(time_points) - 0.5
    square_values = np.full(num_samples, 0.5)

    random_sequence = []
    y_true = []

    # Generate the random sequence by randomly choosing between sine and square values
    for _ in range(num_waves):
        if np.random.rand() < 0.5:
            random_sequence.extend(sine_values)
            y_true.append([0]*num_samples)
        else:
            random_sequence.extend(square_values)
            y_true.append([1]*num_samples)

    wave = np.array(random_sequence)
    wave = wave.reshape(-1,1)

    y_true = np.array(y_true)
    y_true = y_true.flatten()

    wave_repeat = np.repeat(wave, nodes, axis=1)

    return (wave, wave_repeat,y_true)

def PAM4(num_samples, nodes):
    SNR = 32
    pam4_values = [-3, -1, 1, 3]
    
    d = [random.choice(pam4_values) for _ in range(num_samples)]
    d = np.array(d)
    print(type(d))

    q = np.zeros_like(d)

    # Compute q(n) based on the given equation
    for n in range(len(d)):
        q[n] = (
            0.08 * (d[n + 2] if n + 2 < len(d) else 0) +
            -0.12 * (d[n + 1] if n + 1 < len(d) else 0) +
            1.0 * d[n] +
            0.18 * (d[n - 1] if n - 1 >= 0 else 0) +
            -0.1 * (d[n - 2] if n - 2 >= 0 else 0) +
            0.091 * (d[n - 3] if n - 3 >= 0 else 0) +
            -0.05 * (d[n - 4] if n - 4 >= 0 else 0) +
            0.04 * (d[n - 5] if n - 5 >= 0 else 0) +
            0.03 * (d[n - 6] if n - 6 >= 0 else 0) +
            0.01 * (d[n - 7] if n - 7 >= 0 else 0)
        )

    u = q + 0.036 * q**2 - 0.011 * q**3

    noisy = u + add_noise(u,SNR)
    noisy = noisy.reshape(-1,1)
    y_true = d

    noisy_repeat = np.repeat(noisy, nodes, axis=1)

    return (noisy, noisy_repeat,y_true)

def masking(wave_repeat, alpha, beta, phi, k):
    # Create a mask 
    np.random.seed(42)
    nodes = wave_repeat.shape[1]

    mask =  (np.random.uniform(-1, 1, nodes)).reshape(1,nodes)
    masked_input = wave_repeat * mask
    
    X = np.zeros_like(masked_input)
    for n in range(wave_repeat.shape[0]):
        for i in range(nodes):
            if k <= i < nodes:
                X[n, i] = np.sin(alpha * X[n-1, i-k] + beta * masked_input[n,i] + phi)
            elif 0 <= i < k:
                X[n, i] = np.sin(alpha * X[n-2, nodes+i-k] + beta * masked_input[n,i] + phi)

    return (X)
        
def add_noise(q, snr_db):
    signal_power = np.mean(q ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(q))
    return q + noise

def classify_pam4(predictions):
    # PAM-4 levels and find the nearest PAM-4 level for each prediction
    levels = np.array([-3, -1, 1, 3])
    classified = np.array([levels[np.argmin(np.abs(levels - p))] for p in predictions])
    return classified