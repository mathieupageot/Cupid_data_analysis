import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def extract_calibration_peaks(spectrum, peak_energies):
    # Find peaks in the spectrum
    peaks, _ = find_peaks(spectrum)

    # Calculate the difference between peak energies and spectrum values
    energy_diff = np.abs(peak_energies[:, np.newaxis] - spectrum[peaks])

    # Find the best match for each peak
    best_matches = np.argmin(energy_diff, axis=0)

    # Extract the peak energies and corresponding spectrum values
    selected_peaks = peaks[best_matches]
    selected_peak_energies = peak_energies[best_matches]

    return selected_peaks, selected_peak_energies

def perform_calibration(spectrum, peak_energies):
    # Extract the selected peaks and their corresponding energies
    selected_peaks, selected_peak_energies = extract_calibration_peaks(spectrum, peak_energies)

    # Perform linear regression to find the calibration coefficient
    calibration_coefficient = np.polyfit(selected_peaks, selected_peak_energies, deg=1)[0]

    return calibration_coefficient

# Example usage
# Example usage
x = np.linspace(0, 100, 1000)  # Energy range
spectrum = np.zeros_like(x)

# Define peak positions and heights
peak_positions = [20, 35, 45, 60, 75, 90]
peak_heights = [0.5, 1.0, 0.8, 1.5, 1.2, 0.7]

# Add Gaussian-shaped peaks to the spectrum
for position, height in zip(peak_positions, peak_heights):
    spectrum += height * np.exp(-(x - position) ** 2 / (2 * 1.5 ** 2))
peak_energies = np.array([2.0, 5.0, 9.0])

# Perform calibration
calibration_coefficient = perform_calibration(spectrum, peak_energies)

# Create subplots for the fit and the spectrum
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot the spectrum
ax1.plot(spectrum, 'b.', label='Spectrum')
ax1.plot(extract_calibration_peaks(spectrum, peak_energies)[0], spectrum[extract_calibration_peaks(spectrum, peak_energies)[0]], 'ro', label='Selected Peaks')
ax1.set_ylabel('Arbitrary Unit')
ax1.legend()

# Plot the fit
x = np.arange(len(spectrum))
y_fit = calibration_coefficient * x
ax2.plot(x, y_fit, 'r-', label='Fit')
ax2.plot(extract_calibration_peaks(spectrum, peak_energies)[0], extract_calibration_peaks(spectrum, peak_energies)[1], 'bo', label='Selected Peaks')
ax2.set_xlabel('Index')
ax2.set_ylabel('Energy Unit')
ax2.legend()

plt.tight_layout()
plt.show()

