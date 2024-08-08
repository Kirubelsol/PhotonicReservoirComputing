import numpy as np

def generate_half_sine_wave(t, period_seconds):
    frequency = 1 / (2 * period_seconds)  # Half the frequency so that one half-cycle fits the entire period
    sine_wave = np.sin(2 * np.pi * frequency * t)
    half_sine_wave = 2*sine_wave
    return half_sine_wave - 1

def generate_half_square_wave(t, period_seconds):
    return 1*np.ones_like(t)

def sample_and_hold_concatenated_wave(concatenated_wave, total_samples, hold_samples, total_waves, samples_per_period):
    hold_period = int(total_samples / (total_waves * hold_samples))
    sampled_wave = np.zeros_like(concatenated_wave)
    for i in range(total_waves * hold_samples):
        start_idx = i * hold_period
        end_idx = min((i + 1) * hold_period, total_samples)
        sampled_wave[start_idx:end_idx] = concatenated_wave[start_idx]
    return sampled_wave

def generate_random_sampled_wave(total_samples, hold_samples, N, total_waves, samples_per_period, method):

    if method == 'six':
        values = [0.99, 0.79, 0.59, 0.39, 0.19, 0.01]
        random_values = np.random.choice(values, N)
    elif method == 'rand':
        random_values = np.random.rand(N)
    else:
        raise ValueError("Invalid method. Use 'x' or 'y'.")
    
    hold_period = int(total_samples / (total_waves * hold_samples))
    sampled_wave = np.zeros(total_samples)

    for i in range(total_waves * hold_samples):
        start_idx = i * hold_period
        end_idx = min((i + 1) * hold_period, total_samples)
        sample_and_hold_period = hold_period // N
        for j in range(N):
            sample_start_idx = start_idx + j * sample_and_hold_period
            sample_end_idx = min(sample_start_idx + sample_and_hold_period, end_idx)
            sampled_wave[sample_start_idx:sample_end_idx] = random_values[j]
    return sampled_wave



# Function to quantize classification_wave to the closest discrete value
def quantize_to_discrete_values(classification_wave):
    discrete_values = np.linspace(-1, 1, 255)
    
    indices = np.digitize(classification_wave, discrete_values) - 1
    indices = np.clip(indices, 0, len(discrete_values) - 1)
    quantized_values = discrete_values[indices]

    return quantized_values
