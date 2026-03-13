import numpy as np
import matplotlib.pyplot as plt

def add_noise(audio, noise_level=0.005):
    """
    Adds Gaussian noise to an audio signal.

    Parameters:
        audio (np.ndarray): Input audio signal as a numpy array.
        noise_level (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Noisy audio signal.
    """
    noise = np.random.normal(0, noise_level, audio.shape)
    noisy_audio = audio + noise
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
    
    # Add noise
    noise_level = 0.01  # Adjust as needed
    corrupted_data = add_noise(original_data, noise_level)
    
    # Plot original and corrupted data
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t[:1000], original_data[:1000])  # Plot first 1000 samples
    plt.title('Original Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 1, 2)
    plt.plot(t[:1000], corrupted_data[:1000])
    plt.title('Corrupted Data (with Static Noise)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    # Save corrupted data to file (optional)
    np.save('corrupted_data.npy', corrupted_data)
    print("Corrupted data saved to 'corrupted_data.npy'")