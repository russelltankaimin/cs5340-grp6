import numpy as np
import matplotlib.pyplot as plt

def add_noise(audio, noise_level=0.005, noise_type='gaussian', start_time=None, end_time=None, sample_rate=44100):
    """
    Adds various types of noise to an audio signal, optionally within a specified time range.

    Parameters:
        audio (np.ndarray): Input audio signal as a numpy array.
        noise_level (float): Intensity of the noise (interpretation depends on noise_type).
        noise_type (str): Type of noise ('gaussian', 'uniform', 'salt_pepper', 'impulsive').
        start_time (float, optional): Start time in seconds to add noise. If None, noise added to entire signal.
        end_time (float, optional): End time in seconds to add noise. If None, noise added to entire signal.
        sample_rate (int): Sample rate of the audio signal.

    Returns:
        np.ndarray: Noisy audio signal.
    """
    # Determine the range to add noise
    if start_time is not None and end_time is not None:
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        start_idx = max(0, start_idx)
        end_idx = min(len(audio), end_idx)
    else:
        start_idx = 0
        end_idx = len(audio)
    
    # Generate noise based on type
    segment_length = end_idx - start_idx
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, segment_length)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level, noise_level, segment_length)
    elif noise_type == 'salt_pepper':
        # Salt and pepper: random values at -1 or 1 with probability noise_level
        noise = np.zeros(segment_length)
        mask = np.random.random(segment_length) < noise_level
        noise[mask] = np.random.choice([-1, 1], size=np.sum(mask))
    elif noise_type == 'impulsive':
        # Impulsive noise: occasional large spikes
        noise = np.zeros(segment_length)
        spike_indices = np.random.random(segment_length) < noise_level
        noise[spike_indices] = np.random.normal(0, 1, np.sum(spike_indices))  # Large variance for spikes
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    # Create full noise array
    full_noise = np.zeros_like(audio)
    full_noise[start_idx:end_idx] = noise
    
    # Add noise
    noisy_audio = audio + full_noise
    
    # Clip to maintain the same range as the input
    if audio.dtype.kind == 'f':
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    else:
        info = np.iinfo(audio.dtype)
        noisy_audio = np.clip(noisy_audio, info.min, info.max)
    return noisy_audio

if __name__ == "__main__":
    # Example usage: Generate static noise corruption from original data
    # For demonstration, create a sample sine wave as original data
    
    
    # Parameters
    sample_rate = 44100  # Hz
    duration = 2.0  # seconds
    frequency = 440.0  # A4 note
    
    # Generate original data (sine wave)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    original_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Example 1: Add Gaussian noise to entire signal
    noise_level = 0.01
    corrupted_data_gaussian = add_noise(original_data, noise_level, noise_type='gaussian', sample_rate=sample_rate)
    
    # Example 2: Add uniform noise from 0.5 to 1.0 seconds
    corrupted_data_uniform = add_noise(original_data, noise_level, noise_type='uniform', start_time=0.5, end_time=1.0, sample_rate=sample_rate)
    
    # Example 3: Add salt and pepper noise from 1.0 to 1.5 seconds
    corrupted_data_salt_pepper = add_noise(original_data, noise_level=0.05, noise_type='salt_pepper', start_time=1.0, end_time=1.5, sample_rate=sample_rate)
    
    # Example 4: Add impulsive noise from 0.2 to 0.4 seconds
    corrupted_data_impulsive = add_noise(original_data, noise_level=0.02, noise_type='impulsive', start_time=0.2, end_time=0.4, sample_rate=sample_rate)
    
    # Plot original and corrupted data
    plt.figure(figsize=(12, 10))
    
    plt.subplot(5, 1, 1)
    plt.plot(t[:1000], original_data[:1000])  # Plot first 1000 samples
    plt.title('Original Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(5, 1, 2)
    plt.plot(t[:1000], corrupted_data_gaussian[:1000])
    plt.title('Gaussian Noise (Entire Signal)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(5, 1, 3)
    plt.plot(t[:1000], corrupted_data_uniform[:1000])
    plt.title('Uniform Noise (0.5-1.0s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(5, 1, 4)
    plt.plot(t[:1000], corrupted_data_salt_pepper[:1000])
    plt.title('Salt & Pepper Noise (1.0-1.5s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(5, 1, 5)
    plt.plot(t[:1000], corrupted_data_impulsive[:1000])
    plt.title('Impulsive Noise (0.2-0.4s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    # Save one example corrupted data to file (optional)
    np.save('corrupted_data.npy', corrupted_data_gaussian)
    print("Corrupted data saved to 'corrupted_data.npy'")