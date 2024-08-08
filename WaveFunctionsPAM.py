import numpy as np


def generate_mask(N,pam_samples):

    # Generate samples_period uniformly distributed random values between -1 and 1
    randoms = np.random.uniform(-1, 1, N)
    mask = np.tile(randoms, pam_samples)
    return mask 

def add_noise(q, snr_db):
    signal_power = np.mean(q ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(q))
    return q + noise