import numpy as np
import matplotlib.pyplot as plt
from WaveFunctionsPAM import *
import os
import random

#Prepare the PAM4 Data for the AWG

# Constants
period_microseconds = 0.001838941176 #duration of single node of single value
period_seconds = period_microseconds * 1e-6
sample_rate = 82.244444e9  # in Gsa/S
samples_per_period = int(sample_rate * period_seconds) #actual sample considering the sample_rate
N = 50
pam_samples = 60

# Create a directory
folder_name = "train10"
os.makedirs(folder_name, exist_ok=True) 
SNR = 32 #db

# Generate time array for one period 
t = np.linspace(0, period_seconds, samples_per_period, endpoint=False)

# Possible PAM-4 signal values 
pam4_values = [-3, -1, 1, 3]

# Generate N random samples from pam4_values
d = [random.choice(pam4_values) for _ in range(pam_samples)]

q = np.zeros_like(d)

# Compute q(n) based on the given equation (resembling the non-linear channel on most research papers)
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

# Repeat each sample number of node times 
pam4_sampled = []
for sample in noisy:
    pam4_sampled.extend([sample] * N)

# Generate mask
mask = generate_mask(N,pam_samples)
#Generate the input to the AWG wave
masked_signal = pam4_sampled*mask
print(d)

AWG_wave = []
for x in masked_signal:
    AWG_wave.extend([x] * samples_per_period)

total_samples = samples_per_period * N * pam_samples

t_total = np.linspace(0, period_seconds * pam_samples * N, total_samples, endpoint=False)

#to pad zeros for indicating the start and end
num_zeros = int((523200-total_samples)/2)
padded_AWG = np.pad(AWG_wave, (num_zeros, num_zeros), 'constant', constant_values=(0, 0))
t_padded_stop = (period_seconds * pam_samples * N)*(padded_AWG.size/len(AWG_wave))
t_padded = np.linspace(0, t_padded_stop, padded_AWG.size, endpoint=False)


# Define the file path
file_path_real = os.path.join(folder_name, "real.csv")
file_path_AWG = os.path.join(folder_name, "AWG.csv")
file_path_parameters = os.path.join(folder_name, "parameters.csv")
file_path_padded_AWG = os.path.join(folder_name, "padded_AWG.csv") 

parameters = ['N','sample_rate','single_signal__period_micro','samples_per_period','total_waves','t_total_micro']
parameter_values = [N,sample_rate,period_microseconds,samples_per_period,total_samples,t_total[-1]]
parameter_data = np.column_stack((parameters, parameter_values))

# Save to a CSV file
np.savetxt(file_path_real, d, delimiter=',', comments='',fmt='%.3f')
np.savetxt(file_path_AWG, AWG_wave, delimiter=',', comments='', header='SampleRate = 82.2444 GHz', fmt='%.3f')
np.savetxt(file_path_parameters, parameter_data, fmt='%s', delimiter=',', header='Parameter,Value')
np.savetxt(file_path_padded_AWG, padded_AWG, delimiter=',', header='SampleRate = 82.2444 GHz',comments='',fmt='%.3f')


# #plot the input AWG wave
# plt.figure(figsize=(12, 6))
# plt.plot(t_total * 1e6, AWG_wave)  # Convert time to microseconds for plotting
# plt.title('AWG input wave')
# plt.xlabel('Time (µs)')
# plt.ylabel('Amplitude')
# plt.grid(True)


# #plot the padded AWG wave
# plt.figure(figsize=(12, 6))
# plt.plot(t_padded * 1e6, padded_AWG)  # Convert time to microseconds for plotting
# plt.title('Padded AWG')
# plt.xlabel('Time (µs)')
# plt.ylabel('Amplitude')
# plt.grid(True)


# #plot the repeated mask
# x_values = np.arange(len(mask))
# plt.figure(figsize=(12, 6))
# plt.plot(x_values, mask)
# plt.title('mask')
# plt.xlabel('Sample Index')
# plt.ylabel('mask')
# plt.grid(True)

# #plot the repeated pam 4
# x_values = np.arange(len(pam4_sampled))
# plt.figure(figsize=(12, 6))
# plt.plot(x_values, pam4_sampled)
# plt.title('Repeated Pam4 noisy')
# plt.xlabel('Sample Index')
# plt.ylabel('Class')
# plt.grid(True)



# #plot the real y
# x_values = np.arange(len(d))
# plt.figure(figsize=(12, 6))
# plt.plot(x_values, d)
# plt.title('original pam4')
# plt.xlabel('Sample Index')
# plt.ylabel('Class')
# plt.grid(True)
# plt.show()

