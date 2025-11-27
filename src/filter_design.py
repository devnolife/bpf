"""
Filter Design Module
====================
Core functions for Band Pass Filter design using Butterworth approximation.

Contains:
- design_butterworth_bpf: Complete BPF design using scipy
- manual_butterworth_poles: Calculate Butterworth poles manually
- lpf_to_bpf_transform: LPF to BPF frequency transformation
- bilinear_transform: Analog to digital conversion
- get_sos_representation: Get second-order sections
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Any


def calculate_adjusted_cutoffs(fl: float, fu: float, k1_db: float, 
                                order: int, fs: float) -> Tuple[float, float]:
    """
    Calculate adjusted cutoff frequencies to meet passband ripple specification.
    
    Butterworth filter has -3dB at cutoff. To have at most -k1 dB at 
    passband edges (fl and fu), the actual cutoff frequencies need to be
    OUTSIDE the specified passband (lower cutoff < fl, upper cutoff > fu).
    
    Parameters:
    -----------
    fl : float
        Desired lower passband edge in Hz (should have ≤ k1_db attenuation)
    fu : float
        Desired upper passband edge in Hz (should have ≤ k1_db attenuation)
    k1_db : float
        Maximum passband attenuation in dB
    order : int
        Filter order (prototype LPF)
    fs : float
        Sampling frequency in Hz
    
    Returns:
    --------
    tuple
        (fl_adj, fu_adj) - Adjusted cutoff frequencies for filter design
    """
    # For Butterworth: |H(jω)|² = 1 / (1 + (ω/ωc)^(2n))
    # At -3dB (cutoff): |H|² = 0.5, so (ω/ωc)^(2n) = 1
    # At -k1 dB: |H|² = 10^(-k1/10)
    #   10^(-k1/10) = 1 / (1 + (ω/ωc)^(2n))
    #   (ω/ωc)^(2n) = 10^(k1/10) - 1
    #   ω/ωc = (10^(k1/10) - 1)^(1/(2n))
    
    # Since k1 < 3dB, at the passband edge we're INSIDE the -3dB point
    # So ω_edge < ωc for low cutoff (ω_edge = fl, ωc = fl_adj)
    # And ω_edge > ωc for high cutoff (need to swap sign)
    
    epsilon_sq = 10**(k1_db/10) - 1
    # Ratio: ω_edge/ωc for -k1 dB point (< 1 since k1 < 3)
    ratio = epsilon_sq**(1/(2*order))
    
    # For lower cutoff: fl is at -k1 dB, so fl/fl_adj = ratio
    # fl_adj = fl / ratio (fl_adj < fl won't work - we need fl_adj < fl for BPF)
    # Actually for BPF lower edge, to have -k1 dB at fl, cutoff must be BELOW fl
    # fl_adj / fl = ratio => fl_adj = fl * ratio (this makes fl_adj < fl since ratio < 1)
    
    # For upper cutoff: fu is at -k1 dB, so fu/fu_adj = ratio  
    # fu_adj = fu / ratio (this makes fu_adj > fu since ratio < 1)
    
    fl_adj = fl * ratio  # Lower cutoff below fl
    fu_adj = fu / ratio  # Upper cutoff above fu
    
    return fl_adj, fu_adj


def design_butterworth_bpf(order: int, Wn_low: float, Wn_high: float, 
                           fs: float, adjust_for_passband: bool = False,
                           k1_db: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth Band Pass Filter using scipy.
    
    Parameters:
    -----------
    order : int
        Filter order (for the prototype LPF, actual BPF order is 2n)
    Wn_low : float
        Lower cutoff frequency in Hz
    Wn_high : float
        Upper cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    adjust_for_passband : bool
        If True, adjust cutoffs to meet passband ripple spec
    k1_db : float
        Passband ripple specification in dB (used if adjust_for_passband=True)
    
    Returns:
    --------
    tuple
        (b, a) - Numerator and denominator coefficients of H(z)
    
    Example:
    --------
    >>> b, a = design_butterworth_bpf(4, 920, 1040, 6000)
    """
    # Optionally adjust cutoffs to meet passband spec
    if adjust_for_passband:
        Wn_low, Wn_high = calculate_adjusted_cutoffs(Wn_low, Wn_high, k1_db, order, fs)
    
    # Normalize frequencies to Nyquist
    nyquist = fs / 2
    Wn = [Wn_low / nyquist, Wn_high / nyquist]
    
    # Design filter
    b, a = signal.butter(order, Wn, btype='band', analog=False)
    
    return b, a


def design_butterworth_bpf_sos(order: int, Wn_low: float, Wn_high: float, 
                                fs: float, adjust_for_passband: bool = False,
                                k1_db: float = 2.0) -> np.ndarray:
    """
    Design a Butterworth BPF and return second-order sections.
    
    SOS representation is more numerically stable for higher orders.
    
    Parameters:
    -----------
    order : int
        Filter order
    Wn_low : float
        Lower cutoff frequency in Hz
    Wn_high : float
        Upper cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    adjust_for_passband : bool
        If True, adjust cutoffs to meet passband ripple spec
    k1_db : float
        Passband ripple specification in dB
    
    Returns:
    --------
    ndarray
        Second-order sections representation
    """
    # Optionally adjust cutoffs to meet passband spec
    if adjust_for_passband:
        Wn_low, Wn_high = calculate_adjusted_cutoffs(Wn_low, Wn_high, k1_db, order, fs)
    
    nyquist = fs / 2
    Wn = [Wn_low / nyquist, Wn_high / nyquist]
    
    sos = signal.butter(order, Wn, btype='band', analog=False, output='sos')
    
    return sos


def design_analog_butterworth_lpf(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a normalized analog Butterworth lowpass filter.
    
    Creates prototype LPF with cutoff at ω = 1 rad/s.
    
    Parameters:
    -----------
    order : int
        Filter order
    
    Returns:
    --------
    tuple
        (z, p, k) - Zeros, poles, and gain of the analog prototype
    
    Note:
    -----
    Butterworth poles are located at:
    s_k = exp(j * π * (2k + n - 1) / (2n)) for k = 1, 2, ..., n
    """
    # Use scipy for prototype
    z, p, k = signal.buttap(order)
    
    return z, p, k


def design_analog_butterworth_bpf(order: int, Omega_l: float, 
                                   Omega_u: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Design an analog Butterworth bandpass filter.
    
    Parameters:
    -----------
    order : int
        Filter order of prototype LPF
    Omega_l : float
        Lower cutoff frequency (rad/s)
    Omega_u : float
        Upper cutoff frequency (rad/s)
    
    Returns:
    --------
    tuple
        (z, p, k) - Zeros, poles, and gain
    """
    # Get prototype LPF
    z_lp, p_lp, k_lp = signal.buttap(order)
    
    # Transform to bandpass
    Omega_0 = np.sqrt(Omega_l * Omega_u)
    B = Omega_u - Omega_l
    
    z_bp, p_bp, k_bp = signal.lp2bp_zpk(z_lp, p_lp, k_lp, wo=Omega_0, bw=B)
    
    return z_bp, p_bp, k_bp


def analog_to_digital_bilinear(b_analog: np.ndarray, a_analog: np.ndarray, 
                                fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert analog filter to digital using bilinear transformation.
    
    Parameters:
    -----------
    b_analog : ndarray
        Numerator coefficients of analog filter
    a_analog : ndarray
        Denominator coefficients of analog filter
    fs : float
        Sampling frequency
    
    Returns:
    --------
    tuple
        (b_digital, a_digital) - Digital filter coefficients
    
    Note:
    -----
    Bilinear transformation: s = (2/T) * (z-1)/(z+1)
    """
    b_digital, a_digital = signal.bilinear(b_analog, a_analog, fs)
    
    return b_digital, a_digital


def zpk_to_tf(z: np.ndarray, p: np.ndarray, k: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert zero-pole-gain representation to transfer function.
    
    Parameters:
    -----------
    z : ndarray
        Zeros
    p : ndarray
        Poles
    k : float
        Gain
    
    Returns:
    --------
    tuple
        (b, a) - Numerator and denominator coefficients
    """
    b, a = signal.zpk2tf(z, p, k)
    return b, a


def get_butterworth_poles(order: int) -> np.ndarray:
    """
    Calculate Butterworth poles for normalized LPF.
    
    Parameters:
    -----------
    order : int
        Filter order
    
    Returns:
    --------
    ndarray
        Complex poles on the unit circle left half-plane
    
    Formula:
    --------
    s_k = exp(j * π * (2k + n - 1) / (2n)) for k = 1, 2, ..., n
    
    Only poles in the left half-plane (LHP) are used for stability.
    """
    poles = []
    for k in range(1, order + 1):
        theta = np.pi * (2 * k + order - 1) / (2 * order)
        pole = np.exp(1j * theta)
        poles.append(pole)
    
    return np.array(poles)


def manual_bpf_design(order: int, fl: float, fu: float, fs: float, 
                      k1_db: float = 2.0) -> Dict[str, Any]:
    """
    Complete manual BPF design with step-by-step calculations.
    
    Parameters:
    -----------
    order : int
        Filter order
    fl : float
        Lower cutoff frequency (Hz)
    fu : float
        Upper cutoff frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    k1_db : float
        Passband ripple in dB
    
    Returns:
    --------
    dict
        Dictionary containing all design parameters and results
    """
    from .filter_helpers import analog_to_digital, prewarping, calculate_bpf_parameters
    
    results = {}
    
    # Step 1: Digital frequencies
    omega_l = analog_to_digital(fl, fs)
    omega_u = analog_to_digital(fu, fs)
    results['omega_l'] = omega_l
    results['omega_u'] = omega_u
    
    # Step 2: Prewarping
    Omega_l = prewarping(omega_l)
    Omega_u = prewarping(omega_u)
    results['Omega_l'] = Omega_l
    results['Omega_u'] = Omega_u
    
    # Step 3: BPF parameters
    Omega_0, B = calculate_bpf_parameters(Omega_l, Omega_u)
    results['Omega_0'] = Omega_0
    results['B'] = B
    
    # Step 4: Design analog BPF
    z_bp, p_bp, k_bp = design_analog_butterworth_bpf(order, Omega_l, Omega_u)
    results['analog_zeros'] = z_bp
    results['analog_poles'] = p_bp
    results['analog_gain'] = k_bp
    
    # Step 5: Convert to transfer function
    b_analog, a_analog = zpk_to_tf(z_bp, p_bp, k_bp)
    results['b_analog'] = b_analog
    results['a_analog'] = a_analog
    
    # Step 6: Bilinear transformation
    b_digital, a_digital = analog_to_digital_bilinear(b_analog, a_analog, fs)
    results['b_digital'] = b_digital
    results['a_digital'] = a_digital
    
    # Step 7: Get digital poles and zeros
    z_digital = np.roots(b_digital)
    p_digital = np.roots(a_digital)
    results['digital_zeros'] = z_digital
    results['digital_poles'] = p_digital
    
    return results


def print_transfer_function(b: np.ndarray, a: np.ndarray, 
                            domain: str = 'z') -> None:
    """
    Print the transfer function in a readable format.
    
    Parameters:
    -----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    domain : str
        'z' for digital, 's' for analog
    """
    print(f"\nTransfer Function H({domain}):")
    print("=" * 60)
    
    # Numerator
    print(f"\nNumerator coefficients (b):")
    for i, coef in enumerate(b):
        if domain == 'z':
            print(f"  b[{i}] = {coef:>15.10f}  (z^{-i})")
        else:
            print(f"  b[{i}] = {coef:>15.10f}  (s^{len(b)-1-i})")
    
    # Denominator
    print(f"\nDenominator coefficients (a):")
    for i, coef in enumerate(a):
        if domain == 'z':
            print(f"  a[{i}] = {coef:>15.10f}  (z^{-i})")
        else:
            print(f"  a[{i}] = {coef:>15.10f}  (s^{len(a)-1-i})")
    
    print("\n" + "=" * 60)


def get_difference_equation(b: np.ndarray, a: np.ndarray) -> str:
    """
    Generate the difference equation from transfer function coefficients.
    
    Parameters:
    -----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    
    Returns:
    --------
    str
        Difference equation as a string
    """
    # Output terms
    y_terms = []
    for i in range(1, len(a)):
        if a[i] != 0:
            sign = '-' if a[i] > 0 else '+'
            y_terms.append(f"{sign} {abs(a[i]):.6f}*y[n-{i}]")
    
    # Input terms
    x_terms = []
    for i, coef in enumerate(b):
        if coef != 0:
            if i == 0:
                x_terms.append(f"{coef:.6f}*x[n]")
            else:
                sign = '+' if coef > 0 else '-'
                x_terms.append(f"{sign} {abs(coef):.6f}*x[n-{i}]")
    
    # Combine
    eq = f"y[n] = " + " ".join(x_terms) + " " + " ".join(y_terms)
    
    return eq


if __name__ == "__main__":
    print("Testing Filter Design Module")
    print("=" * 60)
    
    # Design parameters
    order = 4
    fl = 920    # Hz
    fu = 1040   # Hz
    fs = 6000   # Hz
    
    # Design using scipy
    print(f"\nDesigning Butterworth BPF:")
    print(f"  Order: {order}")
    print(f"  Passband: {fl} - {fu} Hz")
    print(f"  Sampling: {fs} Hz")
    
    b, a = design_butterworth_bpf(order, fl, fu, fs)
    
    print_transfer_function(b, a, 'z')
    
    # Print difference equation
    print("\nDifference Equation:")
    print(get_difference_equation(b, a))
    
    # Get SOS for stability
    sos = design_butterworth_bpf_sos(order, fl, fu, fs)
    print(f"\nSecond-Order Sections (SOS): {sos.shape[0]} sections")
