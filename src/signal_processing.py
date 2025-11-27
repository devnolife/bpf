"""
Signal Processing Module
========================
Signal generation and filtering operations for BPF demonstration.

Contains:
- generate_test_signal: Create multi-frequency test signal
- apply_filter: Filter signal using BPF
- plot_input_output: Compare input and output signals
- fft_analysis: Frequency domain analysis
- plot_time_frequency: Combined time and frequency plots
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional, List
import os


def generate_test_signal(fs: float, duration: float,
                         frequencies: List[float],
                         amplitudes: Optional[List[float]] = None,
                         add_noise: bool = False,
                         noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a multi-frequency test signal.
    
    Parameters:
    -----------
    fs : float
        Sampling frequency (Hz)
    duration : float
        Signal duration (seconds)
    frequencies : list
        List of frequency components (Hz)
    amplitudes : list, optional
        Amplitude for each frequency (default all 1.0)
    add_noise : bool
        Whether to add Gaussian noise
    noise_level : float
        Standard deviation of noise
    
    Returns:
    --------
    tuple
        (t, x) - Time vector and signal
    
    Example:
    --------
    >>> t, x = generate_test_signal(6000, 0.1, [500, 980, 1500])
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    if amplitudes is None:
        amplitudes = [1.0] * len(frequencies)
    
    # Generate signal as sum of sinusoids
    x = np.zeros(n_samples)
    for freq, amp in zip(frequencies, amplitudes):
        x += amp * np.sin(2 * np.pi * freq * t)
    
    # Add noise if requested
    if add_noise:
        x += noise_level * np.random.randn(n_samples)
    
    return t, x


def generate_chirp_signal(fs: float, duration: float,
                          f0: float, f1: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a chirp (frequency sweep) signal.
    
    Parameters:
    -----------
    fs : float
        Sampling frequency (Hz)
    duration : float
        Signal duration (seconds)
    f0 : float
        Start frequency (Hz)
    f1 : float
        End frequency (Hz)
    
    Returns:
    --------
    tuple
        (t, x) - Time vector and chirp signal
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    x = signal.chirp(t, f0, duration, f1, method='linear')
    
    return t, x


def apply_filter(x: np.ndarray, b: np.ndarray, a: np.ndarray,
                 use_filtfilt: bool = True) -> np.ndarray:
    """
    Apply the filter to a signal.
    
    Parameters:
    -----------
    x : ndarray
        Input signal
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    use_filtfilt : bool
        If True, use zero-phase filtering (filtfilt)
        If False, use causal filtering (lfilter)
    
    Returns:
    --------
    ndarray
        Filtered signal
    
    Note:
    -----
    filtfilt applies filter twice (forward and backward) for zero phase
    but doubles the filter order effect.
    """
    if use_filtfilt:
        y = signal.filtfilt(b, a, x)
    else:
        y = signal.lfilter(b, a, x)
    
    return y


def apply_filter_sos(x: np.ndarray, sos: np.ndarray,
                     use_filtfilt: bool = True) -> np.ndarray:
    """
    Apply filter using second-order sections (more stable).
    
    Parameters:
    -----------
    x : ndarray
        Input signal
    sos : ndarray
        Second-order sections
    use_filtfilt : bool
        Whether to use zero-phase filtering
    
    Returns:
    --------
    ndarray
        Filtered signal
    """
    if use_filtfilt:
        y = signal.sosfiltfilt(sos, x)
    else:
        y = signal.sosfilt(sos, x)
    
    return y


def compute_fft(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the FFT of a signal.
    
    Parameters:
    -----------
    x : ndarray
        Input signal
    fs : float
        Sampling frequency
    
    Returns:
    --------
    tuple
        (freq, magnitude) - Frequency vector and magnitude spectrum
    """
    n = len(x)
    freq = fftfreq(n, 1/fs)[:n//2]
    X = fft(x)
    magnitude = 2/n * np.abs(X[:n//2])
    
    return freq, magnitude


def compute_fft_db(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT magnitude in dB.
    
    Parameters:
    -----------
    x : ndarray
        Input signal
    fs : float
        Sampling frequency
    
    Returns:
    --------
    tuple
        (freq, magnitude_db) - Frequency and magnitude in dB
    """
    freq, mag = compute_fft(x, fs)
    mag_db = 20 * np.log10(mag + 1e-12)
    
    return freq, mag_db


def plot_input_output(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                      fs: float, specs: Optional[dict] = None,
                      save_path: Optional[str] = None) -> None:
    """
    Plot input and output signals in time and frequency domains.
    
    Parameters:
    -----------
    t : ndarray
        Time vector
    x : ndarray
        Input signal
    y : ndarray
        Output (filtered) signal
    fs : float
        Sampling frequency
    specs : dict, optional
        Filter specifications for marking
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Limit display samples for clarity
    display_samples = min(len(t), int(0.02 * fs))  # 20ms max
    
    # Input - Time domain
    ax1 = axes[0, 0]
    ax1.plot(t[:display_samples] * 1000, x[:display_samples], 'b-', linewidth=0.8)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Input Signal x(t) - Time Domain')
    ax1.grid(True, alpha=0.3)
    
    # Output - Time domain
    ax2 = axes[0, 1]
    ax2.plot(t[:display_samples] * 1000, y[:display_samples], 'r-', linewidth=0.8)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Output Signal y(t) - Time Domain')
    ax2.grid(True, alpha=0.3)
    
    # Input - Frequency domain
    ax3 = axes[1, 0]
    freq_x, mag_x = compute_fft(x, fs)
    ax3.plot(freq_x, mag_x, 'b-', linewidth=1)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Input Signal X(f) - Frequency Domain')
    ax3.set_xlim([0, fs/2])
    ax3.grid(True, alpha=0.3)
    
    if specs:
        ax3.axvline(x=specs['fl'], color='g', linestyle='--', alpha=0.7, label='Passband')
        ax3.axvline(x=specs['fu'], color='g', linestyle='--', alpha=0.7)
        ax3.axvspan(specs['fl'], specs['fu'], alpha=0.1, color='green')
        ax3.legend()
    
    # Output - Frequency domain
    ax4 = axes[1, 1]
    freq_y, mag_y = compute_fft(y, fs)
    ax4.plot(freq_y, mag_y, 'r-', linewidth=1)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.set_title('Output Signal Y(f) - Frequency Domain')
    ax4.set_xlim([0, fs/2])
    ax4.grid(True, alpha=0.3)
    
    if specs:
        ax4.axvline(x=specs['fl'], color='g', linestyle='--', alpha=0.7, label='Passband')
        ax4.axvline(x=specs['fu'], color='g', linestyle='--', alpha=0.7)
        ax4.axvspan(specs['fl'], specs['fu'], alpha=0.1, color='green')
        ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_frequency_comparison(x: np.ndarray, y: np.ndarray, fs: float,
                              specs: Optional[dict] = None,
                              save_path: Optional[str] = None) -> None:
    """
    Compare input and output spectra on the same plot.
    
    Parameters:
    -----------
    x : ndarray
        Input signal
    y : ndarray
        Output signal
    fs : float
        Sampling frequency
    specs : dict, optional
        Filter specifications
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Linear scale
    freq_x, mag_x = compute_fft(x, fs)
    freq_y, mag_y = compute_fft(y, fs)
    
    ax1.plot(freq_x, mag_x, 'b-', linewidth=1, alpha=0.7, label='Input X(f)')
    ax1.plot(freq_y, mag_y, 'r-', linewidth=1.5, label='Output Y(f)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    ax1.set_title('Spectrum Comparison (Linear Scale)')
    ax1.set_xlim([0, fs/2])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if specs:
        ax1.axvspan(specs['fl'], specs['fu'], alpha=0.1, color='green', label='Passband')
    
    # dB scale
    freq_x, mag_x_db = compute_fft_db(x, fs)
    freq_y, mag_y_db = compute_fft_db(y, fs)
    
    ax2.plot(freq_x, mag_x_db, 'b-', linewidth=1, alpha=0.7, label='Input X(f)')
    ax2.plot(freq_y, mag_y_db, 'r-', linewidth=1.5, label='Output Y(f)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Spectrum Comparison (dB Scale)')
    ax2.set_xlim([0, fs/2])
    ax2.set_ylim([-80, np.max(mag_x_db) + 10])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if specs:
        ax2.axvspan(specs['fl'], specs['fu'], alpha=0.1, color='green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_spectrogram(x: np.ndarray, fs: float, title: str = "Spectrogram",
                     save_path: Optional[str] = None) -> None:
    """
    Plot spectrogram of a signal.
    
    Parameters:
    -----------
    x : ndarray
        Input signal
    fs : float
        Sampling frequency
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    f, t, Sxx = signal.spectrogram(x, fs, nperseg=256)
    
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.set_ylim([0, fs/2])
    
    plt.colorbar(im, ax=ax, label='Power (dB)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def fft_analysis(x: np.ndarray, y: np.ndarray, fs: float,
                 specs: dict, save_dir: Optional[str] = None) -> dict:
    """
    Complete FFT analysis of input and output signals.
    
    Parameters:
    -----------
    x : ndarray
        Input signal
    y : ndarray
        Output signal
    fs : float
        Sampling frequency
    specs : dict
        Filter specifications
    save_dir : str, optional
        Directory to save plots
    
    Returns:
    --------
    dict
        Analysis results
    """
    results = {}
    
    # Compute FFTs
    freq, mag_x = compute_fft(x, fs)
    _, mag_y = compute_fft(y, fs)
    
    # Find peaks in input
    peaks_x, _ = signal.find_peaks(mag_x, height=0.1 * np.max(mag_x))
    results['input_peaks'] = freq[peaks_x]
    results['input_peak_mags'] = mag_x[peaks_x]
    
    # Find peaks in output
    peaks_y, _ = signal.find_peaks(mag_y, height=0.1 * np.max(mag_y))
    results['output_peaks'] = freq[peaks_y]
    results['output_peak_mags'] = mag_y[peaks_y]
    
    # Attenuation at specific frequencies
    def get_attenuation(f_target):
        idx = np.argmin(np.abs(freq - f_target))
        if mag_x[idx] > 0:
            return 20 * np.log10(mag_y[idx] / mag_x[idx])
        return -np.inf
    
    # Check frequencies in stopband
    stopband_freqs = [specs['f1'] * 0.5, specs['f1'], specs['f2'], specs['f2'] * 1.2]
    stopband_freqs = [f for f in stopband_freqs if f < fs/2]
    
    results['stopband_attenuation'] = {}
    for f in stopband_freqs:
        results['stopband_attenuation'][f] = get_attenuation(f)
    
    # Check frequencies in passband
    center_freq = (specs['fl'] + specs['fu']) / 2
    passband_freqs = [specs['fl'], center_freq, specs['fu']]
    
    results['passband_gain'] = {}
    for f in passband_freqs:
        results['passband_gain'][f] = get_attenuation(f)
    
    # Print results
    print("\n" + "=" * 60)
    print("FFT ANALYSIS RESULTS")
    print("=" * 60)
    
    print("\nInput Signal Peaks:")
    for f, m in zip(results['input_peaks'], results['input_peak_mags']):
        print(f"  {f:.1f} Hz: magnitude = {m:.4f}")
    
    print("\nOutput Signal Peaks:")
    for f, m in zip(results['output_peaks'], results['output_peak_mags']):
        print(f"  {f:.1f} Hz: magnitude = {m:.4f}")
    
    print("\nStopband Attenuation:")
    for f, atten in results['stopband_attenuation'].items():
        print(f"  {f:.0f} Hz: {atten:.2f} dB")
    
    print("\nPassband Response:")
    for f, gain in results['passband_gain'].items():
        print(f"  {f:.0f} Hz: {gain:.2f} dB")
    
    print("=" * 60)
    
    # Generate plots
    if save_dir:
        save_path = os.path.join(save_dir, 'frequency_comparison.png')
        plot_frequency_comparison(x, y, fs, specs, save_path)
    else:
        plot_frequency_comparison(x, y, fs, specs)
    
    return results


def demo_filtering(b: np.ndarray, a: np.ndarray, fs: float,
                   specs: dict, save_dir: Optional[str] = None) -> None:
    """
    Complete demonstration of BPF filtering.
    
    Parameters:
    -----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    fs : float
        Sampling frequency
    specs : dict
        Filter specifications
    save_dir : str, optional
        Directory to save results
    """
    print("\n" + "=" * 60)
    print("BAND PASS FILTER DEMONSTRATION")
    print("=" * 60)
    
    # Generate test signal with components in and out of passband
    center_freq = (specs['fl'] + specs['fu']) / 2
    test_freqs = [
        300,           # Below stopband (should be attenuated)
        specs['f1'],   # At lower stopband edge
        center_freq,   # In passband (should pass)
        specs['f2'],   # At upper stopband edge
        2000           # Above stopband (should be attenuated)
    ]
    test_amps = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    print(f"\nTest Signal Frequencies: {test_freqs} Hz")
    print(f"Passband: {specs['fl']} - {specs['fu']} Hz")
    print(f"Expected to pass: ~{center_freq:.0f} Hz")
    print(f"Expected to attenuate: 300, {specs['f1']}, {specs['f2']}, 2000 Hz")
    
    # Generate signal
    duration = 0.1  # 100 ms
    t, x = generate_test_signal(fs, duration, test_freqs, test_amps)
    
    # Apply filter
    y = apply_filter(x, b, a)
    
    # Plot results
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Time/frequency comparison
        save_path = os.path.join(save_dir, 'input_output_comparison.png')
        plot_input_output(t, x, y, fs, specs, save_path)
        
        # FFT analysis
        fft_analysis(x, y, fs, specs, save_dir)
    else:
        plot_input_output(t, x, y, fs, specs)
        fft_analysis(x, y, fs, specs)
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    from filter_design import design_butterworth_bpf
    
    print("Testing Signal Processing Module")
    print("=" * 60)
    
    # Filter specifications
    specs = {
        'f1': 770,    # Hz
        'fl': 920,    # Hz
        'fu': 1040,   # Hz
        'f2': 1155,   # Hz
        'k1': 2,      # dB
        'k2': 40,     # dB
        'fs': 6000    # Hz
    }
    
    # Design filter
    order = 4
    b, a = design_butterworth_bpf(order, specs['fl'], specs['fu'], specs['fs'])
    
    # Run demonstration
    demo_filtering(b, a, specs['fs'], specs)
