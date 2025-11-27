"""
Filter Helper Functions
=======================
Helper functions for Band Pass Filter design calculations.

Contains:
- analog_to_digital: Convert Hz to rad/sample
- prewarping: Digital to analog frequency conversion
- lpf_normalization: BPF to normalized LPF transformation
- calculate_order: Butterworth filter order calculation
"""

import numpy as np


def analog_to_digital(f_hz: float, fs: float) -> float:
    """
    Convert analog frequency (Hz) to digital frequency (rad/sample).
    
    Parameters:
    -----------
    f_hz : float
        Analog frequency in Hz
    fs : float
        Sampling frequency in Hz
    
    Returns:
    --------
    float
        Digital frequency in rad/sample (0 to π)
    
    Formula:
    --------
    ω = 2π * f / fs
    
    Example:
    --------
    >>> analog_to_digital(1000, 8000)
    0.7853981633974483  # π/4 rad/sample
    """
    omega = 2 * np.pi * f_hz / fs
    return omega


def prewarping(omega_digital: float, T: float = 2.0) -> float:
    """
    Apply prewarping to convert digital frequency to analog frequency.
    
    This compensates for frequency warping in bilinear transformation.
    
    Parameters:
    -----------
    omega_digital : float
        Digital frequency in rad/sample
    T : float, optional
        Sampling period scaling factor (default=2 for normalized)
    
    Returns:
    --------
    float
        Prewarped analog frequency in rad/s
    
    Formula:
    --------
    Ω = (2/T) * tan(ω/2)
    
    Example:
    --------
    >>> prewarping(np.pi/4, T=2)
    0.4142135623730951
    """
    Omega = (2 / T) * np.tan(omega_digital / 2)
    return Omega


def calculate_bpf_parameters(Omega_l: float, Omega_u: float) -> tuple:
    """
    Calculate BPF center frequency and bandwidth.
    
    Parameters:
    -----------
    Omega_l : float
        Lower cutoff frequency (rad/s)
    Omega_u : float
        Upper cutoff frequency (rad/s)
    
    Returns:
    --------
    tuple
        (Omega_0, B) - Center frequency and bandwidth
    
    Formulas:
    ---------
    Ω₀ = √(Ω_l * Ω_u)  (geometric mean - center frequency)
    B = Ω_u - Ω_l       (bandwidth)
    """
    Omega_0 = np.sqrt(Omega_l * Omega_u)
    B = Omega_u - Omega_l
    return Omega_0, B


def lpf_normalization(Omega: float, Omega_0: float, B: float) -> float:
    """
    Transform BPF frequency to normalized LPF frequency.
    
    Used to convert band pass specifications to low pass for order calculation.
    
    Parameters:
    -----------
    Omega : float
        BPF frequency (rad/s)
    Omega_0 : float
        BPF center frequency (rad/s)
    B : float
        BPF bandwidth (rad/s)
    
    Returns:
    --------
    float
        Normalized LPF frequency (Ω_s)
    
    Formula:
    --------
    Ω_s = (1/B) * |Ω² - Ω₀²| / Ω
    
    Note:
    -----
    For Butterworth BPF, we need the stopband edge that gives
    the SMALLER normalized frequency (more restrictive).
    """
    Omega_s = (1 / B) * np.abs(Omega**2 - Omega_0**2) / Omega
    return Omega_s


def calculate_order(Omega_s: float, k1_db: float, k2_db: float) -> int:
    """
    Calculate Butterworth filter order.
    
    Parameters:
    -----------
    Omega_s : float
        Normalized stopband edge frequency
    k1_db : float
        Passband ripple in dB
    k2_db : float
        Stopband attenuation in dB
    
    Returns:
    --------
    int
        Minimum filter order (rounded up)
    
    Formula:
    --------
    n ≥ log[(10^(k2/10) - 1) / (10^(k1/10) - 1)] / [2 * log(Ω_s)]
    
    For Butterworth:
    - At passband edge: |H(jΩ)|² = 1 / (1 + ε²) where ε² = 10^(k1/10) - 1
    - At stopband edge: |H(jΩ)|² = 1 / (1 + A²) where A² = 10^(k2/10) - 1
    """
    epsilon_sq = 10**(k1_db / 10) - 1  # Passband ripple parameter
    A_sq = 10**(k2_db / 10) - 1        # Stopband attenuation parameter
    
    n = np.log10(A_sq / epsilon_sq) / (2 * np.log10(Omega_s))
    
    return int(np.ceil(n))


def calculate_cutoff_frequency(k1_db: float) -> float:
    """
    Calculate the 3dB cutoff frequency scaling factor for Butterworth filter.
    
    Parameters:
    -----------
    k1_db : float
        Passband ripple in dB
    
    Returns:
    --------
    float
        Scaling factor for cutoff frequency
    
    Formula:
    --------
    Ω_c = (10^(k1/10) - 1)^(1/2n) for order n
    
    For normalized LPF with passband edge at Ω=1:
    Ω_c = ε^(1/n) where ε = √(10^(k1/10) - 1)
    """
    epsilon = np.sqrt(10**(k1_db / 10) - 1)
    return epsilon


def print_frequency_summary(specs: dict, omega: dict, Omega: dict, 
                           Omega_0: float, B: float) -> None:
    """
    Print a summary of all frequency conversions.
    
    Parameters:
    -----------
    specs : dict
        Original specifications (f1, fl, fu, f2, fs)
    omega : dict
        Digital frequencies (rad/sample)
    Omega : dict
        Prewarped analog frequencies (rad/s)
    Omega_0 : float
        BPF center frequency
    B : float
        BPF bandwidth
    """
    print("=" * 60)
    print("FREQUENCY CONVERSION SUMMARY")
    print("=" * 60)
    
    print("\n1. Original Specifications (Hz):")
    print(f"   f1 (stopband low)  = {specs['f1']:>8.2f} Hz")
    print(f"   fl (passband low)  = {specs['fl']:>8.2f} Hz")
    print(f"   fu (passband high) = {specs['fu']:>8.2f} Hz")
    print(f"   f2 (stopband high) = {specs['f2']:>8.2f} Hz")
    print(f"   fs (sampling)      = {specs['fs']:>8.2f} Hz")
    
    print("\n2. Digital Frequencies (rad/sample):")
    print(f"   ω1 = {omega['omega1']:.6f}")
    print(f"   ωl = {omega['omega_l']:.6f}")
    print(f"   ωu = {omega['omega_u']:.6f}")
    print(f"   ω2 = {omega['omega2']:.6f}")
    
    print("\n3. Prewarped Analog Frequencies (rad/s):")
    print(f"   Ω1 = {Omega['Omega1']:.6f}")
    print(f"   Ωl = {Omega['Omega_l']:.6f}")
    print(f"   Ωu = {Omega['Omega_u']:.6f}")
    print(f"   Ω2 = {Omega['Omega2']:.6f}")
    
    print("\n4. BPF Parameters:")
    print(f"   Center frequency (Ω₀) = {Omega_0:.6f} rad/s")
    print(f"   Bandwidth (B)         = {B:.6f} rad/s")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test with given specifications
    print("Testing Filter Helper Functions")
    print("-" * 40)
    
    # Given specifications
    f1 = 770    # Hz (lower stopband)
    fl = 920    # Hz (lower cutoff)
    fu = 1040   # Hz (upper cutoff)
    f2 = 1155   # Hz (upper stopband)
    fs = 6000   # Hz (sampling)
    k1 = 2      # dB (passband ripple)
    k2 = 40     # dB (stopband attenuation)
    
    # Step 1: Convert to digital frequencies
    omega1 = analog_to_digital(f1, fs)
    omega_l = analog_to_digital(fl, fs)
    omega_u = analog_to_digital(fu, fs)
    omega2 = analog_to_digital(f2, fs)
    
    print(f"\nDigital frequencies (rad/sample):")
    print(f"ω1 = {omega1:.6f}, ωl = {omega_l:.6f}")
    print(f"ωu = {omega_u:.6f}, ω2 = {omega2:.6f}")
    
    # Step 2: Prewarping
    Omega1 = prewarping(omega1)
    Omega_l = prewarping(omega_l)
    Omega_u = prewarping(omega_u)
    Omega2 = prewarping(omega2)
    
    print(f"\nPrewarped frequencies (rad/s):")
    print(f"Ω1 = {Omega1:.6f}, Ωl = {Omega_l:.6f}")
    print(f"Ωu = {Omega_u:.6f}, Ω2 = {Omega2:.6f}")
    
    # Step 3: BPF parameters
    Omega_0, B = calculate_bpf_parameters(Omega_l, Omega_u)
    print(f"\nBPF parameters:")
    print(f"Center frequency Ω₀ = {Omega_0:.6f}")
    print(f"Bandwidth B = {B:.6f}")
    
    # Step 4: LPF normalization
    Omega_s1 = lpf_normalization(Omega1, Omega_0, B)
    Omega_s2 = lpf_normalization(Omega2, Omega_0, B)
    Omega_s = min(Omega_s1, Omega_s2)  # More restrictive
    
    print(f"\nNormalized stopband frequencies:")
    print(f"Ωs1 = {Omega_s1:.6f}, Ωs2 = {Omega_s2:.6f}")
    print(f"Ωs (min) = {Omega_s:.6f}")
    
    # Step 5: Calculate order
    n = calculate_order(Omega_s, k1, k2)
    print(f"\nFilter order n = {n}")
