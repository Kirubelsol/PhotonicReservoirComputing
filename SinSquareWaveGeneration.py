import numpy as np
import matplotlib.pyplot as plt
from WaveFunctionsSinSquare import *
import os

# Constants
period_microseconds = 0.5571445545
period_seconds = period_microseconds * 1e-6
sample_rate = 82.244444e9  # Gsa/s
samples_per_period = int(sample_rate * period_seconds) 
total_waves = 11
N = 100
hold_samples = 6

# Create a directory
folder_name = "11hold_N100_six_test3"
os.makedirs(folder_name, exist_ok=True)
method = 'six'
np.random.seed(67) 

# Generate time array for one period
t = np.linspace(0, period_seconds, samples_per_period, endpoint=False)

# Generate random sine and half-square waves
original_waves = []
classification_wave = []

for i in range(total_waves):
    if i == 0 or i == total_waves - 1:
        # Ensure the first and last wave are square waves
        wave = generate_half_square_wave(t, period_seconds)
        class_wave = np.ones(hold_samples)
    else:
        if np.random.rand() >= 0.5:
            wave = generate_half_sine_wave(t, period_seconds)
            class_wave = np.zeros(hold_samples)
        else:
            wave = generate_half_square_wave(t, period_seconds)
            class_wave = np.ones(hold_samples)
    original_waves.append(wave)
    classification_wave.append(class_wave)

# Concatenate all waves
concatenated_original_wave = np.concatenate(original_waves)
concatenated_classification_wave = np.concatenate(classification_wave)

# Generate time array for the concatenated wave 
total_samples = samples_per_period * total_waves
t_total = np.linspace(0, period_seconds * total_waves, total_samples, endpoint=False)

# Apply sample-and-hold to the concatenated wave
sampled_concatenated_wave = sample_and_hold_concatenated_wave(concatenated_original_wave, total_samples, hold_samples, total_waves, samples_per_period)

# Generate random sampled wave
random_sampled_wave = generate_random_sampled_wave(total_samples, hold_samples, N, total_waves, samples_per_period, method)

#Generate the input to the AWG wave
AWG_wave = sampled_concatenated_wave * random_sampled_wave

#to pad zeros 
num_zeros = int((523200-total_samples)/2)
padded_AWG = np.pad(AWG_wave, (num_zeros, num_zeros), 'constant', constant_values=(0, 0))
t_padded_stop = (period_seconds * total_waves)*(padded_AWG.size/AWG_wave.size)
t_padded = np.linspace(0, t_padded_stop, padded_AWG.size, endpoint=False)

#Combine all variables into a single array for saving
data_all = np.column_stack((AWG_wave, random_sampled_wave, sampled_concatenated_wave, concatenated_original_wave, t_total*10e6))
header_all = "AWG,random,sampled,original,time"

# Define the file path
file_path_all = os.path.join(folder_name, "all_wave_data.csv")
file_path_real = os.path.join(folder_name, "real.csv")
file_path_AWG = os.path.join(folder_name, "AWG.csv")
file_path_parameters = os.path.join(folder_name, "parameters.csv")
file_path_original = os.path.join(folder_name, "original_wave.csv")
file_path_padded_AWG = os.path.join(folder_name, "padded_AWG.csv") 

parameters = ['N','sample_rate','single_wave_period_micro','samples_per_period','total_waves','hold_samples','t_total_micro']
parameter_values = [N,sample_rate,period_microseconds,samples_per_period,total_samples,hold_samples,t_total[-1]]
parameter_data = np.column_stack((parameters, parameter_values))

# Save to a CSV file
np.savetxt(file_path_all, data_all, header=header_all, delimiter=',', comments='')
np.savetxt(file_path_real, concatenated_classification_wave, delimiter=',', comments='',fmt='%.3f')
np.savetxt(file_path_AWG, AWG_wave, delimiter=',', comments='', header='SampleRate = 82.2444 GHz', fmt='%.3f')
np.savetxt(file_path_parameters, parameter_data, fmt='%s', delimiter=',', header='Parameter,Value')
np.savetxt(file_path_original, concatenated_original_wave, delimiter=',', header='SampleRate = 82.2444 GHz',comments='',fmt='%.3f')
np.savetxt(file_path_padded_AWG, padded_AWG, delimiter=',', header='SampleRate = 82.2444 GHz',comments='',fmt='%.3f')

print('yes')

# Plot the concatenated wave
plt.figure(figsize=(12, 6))
plt.plot(t_total * 1e6, concatenated_original_wave)  # Convert time to microseconds for plotting
plt.title('Concatenated Half Sine and Half Square Waves (Amplitude 1)')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot the concatenated sampled wave
plt.figure(figsize=(12, 6))
plt.plot(t_total * 1e6, sampled_concatenated_wave)  # Convert time to microseconds for plotting
plt.title('Concatenated and Sampled Half Sine and Half Square Waves (Amplitude 1)')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot the random sampled wave
plt.figure(figsize=(12, 6))
plt.plot(t_total * 1e6, random_sampled_wave)  # Convert time to microseconds for plotting
plt.title('Random Sampled Wave (Amplitude 0 to 1)')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.grid(True)

#plot the input AWG wave
plt.figure(figsize=(12, 6))
plt.plot(t_total * 1e6, AWG_wave)  # Convert time to microseconds for plotting
plt.title('AWG input wave')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.grid(True)

#plot the padded AWG wave
plt.figure(figsize=(12, 6))
plt.plot(t_padded * 1e6, padded_AWG)  # Convert time to microseconds for plotting
plt.title('Padded AWG')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.grid(True)

#plot the real y
x_values = np.arange(len(concatenated_classification_wave))
plt.figure(figsize=(12, 6))
plt.plot(x_values, concatenated_classification_wave)
plt.title('Classification Wave (0 for Sine, 1 for Square)')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.grid(True)
plt.show()

