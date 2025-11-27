"""
Filter Analysis Module
======================
Visualization and analysis tools for Band Pass Filter.

Contains:
- plot_magnitude_response: Magnitude in dB vs frequency
- plot_phase_response: Phase vs frequency
- plot_pole_zero: Pole-zero diagram
- plot_impulse_response: Impulse response h[n]
- plot_step_response: Step response
- plot_group_delay: Group delay vs frequency
- verify_specifications: Check if filter meets specs
- plot_filter_specifications: Plot ideal filter template
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, Optional, List
import os


# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


def plot_magnitude_response(b: np.ndarray, a: np.ndarray, fs: float,
                            worN: int = 8192, 
                            save_path: Optional[str] = None,
                            specs: Optional[dict] = None) -> None:
    """
    Plot the magnitude response of the filter.
    
    Parameters:
    -----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    fs : float
        Sampling frequency in Hz
    worN : int
        Number of frequency points
    save_path : str, optional
        Path to save the figure
    specs : dict, optional
        Filter specifications for overlay
    """
    # Calculate frequency response
    w, h = signal.freqz(b, a, worN=worN)
    freq = w * fs / (2 * np.pi)
    
    # Magnitude in dB
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Full range plot
    ax1.plot(freq, mag_db, 'b-', linewidth=1.5, label='Magnitude Response')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Band Pass Filter - Magnitude Response')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, fs/2])
    ax1.set_ylim([-80, 5])
    
    # Add specification lines if provided
    if specs:
        # Passband ripple
        ax1.axhline(y=-specs.get('k1', 2), color='g', linestyle='--', 
                   label=f'Passband ripple (-{specs.get("k1", 2)} dB)', alpha=0.7)
        # Stopband attenuation
        ax1.axhline(y=-specs.get('k2', 40), color='r', linestyle='--',
                   label=f'Stopband atten (-{specs.get("k2", 40)} dB)', alpha=0.7)
        # Frequency markers
        for fname, fval in [('f1', specs.get('f1')), ('fl', specs.get('fl')),
                           ('fu', specs.get('fu')), ('f2', specs.get('f2'))]:
            if fval:
                ax1.axvline(x=fval, color='gray', linestyle=':', alpha=0.5)
                ax1.text(fval, -75, fname, ha='center', fontsize=9)
    
    ax1.legend(loc='upper right')
    
    # Zoomed passband plot
    if specs:
        fl, fu = specs.get('fl', 920), specs.get('fu', 1040)
        margin = (fu - fl) * 2
        ax2.plot(freq, mag_db, 'b-', linewidth=1.5)
        ax2.set_xlim([fl - margin, fu + margin])
        ax2.set_ylim([-10, 2])
        ax2.axhline(y=-specs.get('k1', 2), color='g', linestyle='--', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=fl, color='orange', linestyle='--', label='Cutoff frequencies')
        ax2.axvline(x=fu, color='orange', linestyle='--')
    else:
        ax2.plot(freq, mag_db, 'b-', linewidth=1.5)
        ax2.set_xlim([0, fs/2])
        ax2.set_ylim([-10, 2])
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Passband Detail')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_phase_response(b: np.ndarray, a: np.ndarray, fs: float,
                        worN: int = 8192,
                        save_path: Optional[str] = None) -> None:
    """
    Plot the phase response of the filter.
    
    Parameters:
    -----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    fs : float
        Sampling frequency in Hz
    worN : int
        Number of frequency points
    save_path : str, optional
        Path to save the figure
    """
    w, h = signal.freqz(b, a, worN=worN)
    freq = w * fs / (2 * np.pi)
    
    # Unwrap phase
    phase = np.unwrap(np.angle(h))
    phase_deg = np.degrees(phase)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Phase in radians
    ax1.plot(freq, phase, 'b-', linewidth=1.5)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Phase (radians)')
    ax1.set_title('Band Pass Filter - Phase Response')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, fs/2])
    
    # Phase in degrees
    ax2.plot(freq, phase_deg, 'r-', linewidth=1.5)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title('Phase Response (degrees)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, fs/2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_pole_zero(b: np.ndarray, a: np.ndarray,
                   save_path: Optional[str] = None) -> None:
    """
    Plot the pole-zero diagram.
    
    Parameters:
    -----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    save_path : str, optional
        Path to save the figure
    """
    # Get zeros and poles
    zeros = np.roots(b)
    poles = np.roots(a)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit Circle')
    
    # Zeros
    ax.scatter(zeros.real, zeros.imag, marker='o', s=100, 
               facecolors='none', edgecolors='b', linewidths=2, label='Zeros')
    
    # Poles
    ax.scatter(poles.real, poles.imag, marker='x', s=100, 
               c='r', linewidths=2, label='Poles')
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('Pole-Zero Plot')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    
    # Print pole/zero locations
    print("\nPole Locations:")
    for i, p in enumerate(poles):
        print(f"  p{i+1} = {p.real:.6f} + {p.imag:.6f}j  |p| = {np.abs(p):.6f}")
    
    print("\nZero Locations:")
    for i, z in enumerate(zeros):
        print(f"  z{i+1} = {z.real:.6f} + {z.imag:.6f}j  |z| = {np.abs(z):.6f}")
    
    # Check stability
    max_pole_mag = np.max(np.abs(poles))
    if max_pole_mag < 1:
        print(f"\n✓ Filter is STABLE (max|pole| = {max_pole_mag:.6f} < 1)")
    else:
        print(f"\n✗ Filter is UNSTABLE (max|pole| = {max_pole_mag:.6f} >= 1)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_impulse_response(b: np.ndarray, a: np.ndarray, 
                          n_samples: int = 200,
                          save_path: Optional[str] = None) -> None:
    """
    Plot the impulse response h[n].
    
    Parameters:
    -----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    n_samples : int
        Number of samples to compute
    save_path : str, optional
        Path to save the figure
    """
    # Generate impulse
    impulse = np.zeros(n_samples)
    impulse[0] = 1
    
    # Filter response
    h = signal.lfilter(b, a, impulse)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Stem plot
    ax1.stem(range(n_samples), h, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax1.set_xlabel('Sample n')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Impulse Response h[n]')
    ax1.grid(True, alpha=0.3)
    
    # Line plot for envelope
    ax2.plot(range(n_samples), h, 'b-', linewidth=1)
    ax2.fill_between(range(n_samples), h, alpha=0.3)
    ax2.set_xlabel('Sample n')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Impulse Response (Continuous View)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_step_response(b: np.ndarray, a: np.ndarray,
                       n_samples: int = 200,
                       save_path: Optional[str] = None) -> None:
    """
    Plot the step response.
    
    Parameters:
    -----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    n_samples : int
        Number of samples
    save_path : str, optional
        Path to save the figure
    """
    # Generate step input
    step = np.ones(n_samples)
    
    # Filter response
    y = signal.lfilter(b, a, step)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(range(n_samples), y, 'b-', linewidth=1.5, label='Step Response')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Sample n')
    ax.set_ylabel('Amplitude')
    ax.set_title('Step Response')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_group_delay(b: np.ndarray, a: np.ndarray, fs: float,
                     worN: int = 8192,
                     save_path: Optional[str] = None) -> None:
    """
    Plot the group delay.
    
    Parameters:
    -----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    fs : float
        Sampling frequency
    worN : int
        Number of frequency points
    save_path : str, optional
        Path to save the figure
    """
    w, gd = signal.group_delay((b, a), w=worN)
    freq = w * fs / (2 * np.pi)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(freq, gd, 'b-', linewidth=1.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Group Delay (samples)')
    ax.set_title('Group Delay')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, fs/2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_filter_specifications(specs: dict,
                               save_path: Optional[str] = None) -> None:
    """
    Plot the ideal BPF template with specifications.
    
    Parameters:
    -----------
    specs : dict
        Dictionary with keys: f1, fl, fu, f2, k1, k2, fs
    save_path : str, optional
        Path to save the figure
    """
    f1 = specs['f1']   # Lower stopband edge
    fl = specs['fl']   # Lower cutoff
    fu = specs['fu']   # Upper cutoff  
    f2 = specs['f2']   # Upper stopband edge
    k1 = specs['k1']   # Passband ripple (dB)
    k2 = specs['k2']   # Stopband attenuation (dB)
    fs = specs['fs']   # Sampling frequency
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create ideal template
    freqs = [0, f1*0.5, f1, fl, fu, f2, f2*1.2, fs/2]
    mags = [-k2, -k2, -k2, -k1, -k1, -k2, -k2, -k2]
    
    # Fill regions
    # Stopband 1
    ax.fill_between([0, f1], [-100, -100], [-k2, -k2], 
                    color='red', alpha=0.2, label='Stopband')
    # Transition band 1
    ax.fill_between([f1, fl], [-k2, -k2], [-k1, -k1],
                    color='yellow', alpha=0.2, label='Transition')
    # Passband
    ax.fill_between([fl, fu], [0, 0], [-k1, -k1],
                    color='green', alpha=0.2, label='Passband')
    # Transition band 2
    ax.fill_between([fu, f2], [-k1, -k1], [-k2, -k2],
                    color='yellow', alpha=0.2)
    # Stopband 2
    ax.fill_between([f2, fs/2], [-k2, -k2], [-100, -100],
                    color='red', alpha=0.2)
    
    # Draw boundaries
    ax.plot([0, f1, f1], [-k2, -k2, -100], 'r-', linewidth=2)
    ax.plot([f1, fl], [-k2, -k1], 'k--', linewidth=1.5)
    ax.plot([fl, fl, fu, fu], [-100, -k1, -k1, -100], 'g-', linewidth=2)
    ax.plot([fu, f2], [-k1, -k2], 'k--', linewidth=1.5)
    ax.plot([f2, f2, fs/2], [-100, -k2, -k2], 'r-', linewidth=2)
    
    # Annotations
    ax.annotate(f'f1={f1}Hz', xy=(f1, -k2-5), ha='center', fontsize=10)
    ax.annotate(f'fl={fl}Hz', xy=(fl, -k1+3), ha='center', fontsize=10)
    ax.annotate(f'fu={fu}Hz', xy=(fu, -k1+3), ha='center', fontsize=10)
    ax.annotate(f'f2={f2}Hz', xy=(f2, -k2-5), ha='center', fontsize=10)
    
    ax.axhline(y=-k1, color='g', linestyle=':', alpha=0.5)
    ax.axhline(y=-k2, color='r', linestyle=':', alpha=0.5)
    ax.text(fs/2-100, -k1+1, f'-{k1}dB', fontsize=10, color='g')
    ax.text(fs/2-100, -k2+2, f'-{k2}dB', fontsize=10, color='r')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Band Pass Filter Specifications Template')
    ax.set_xlim([0, fs/2])
    ax.set_ylim([-60, 5])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add specification box
    textstr = f'Specifications:\n' \
              f'f1 = {f1} Hz\n' \
              f'fl = {fl} Hz\n' \
              f'fu = {fu} Hz\n' \
              f'f2 = {f2} Hz\n' \
              f'k1 = {k1} dB\n' \
              f'k2 = {k2} dB\n' \
              f'fs = {fs} Hz'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def verify_specifications(b: np.ndarray, a: np.ndarray, 
                         specs: dict, worN: int = 8192) -> dict:
    """
    Verify if the filter meets the specifications.
    
    Parameters:
    -----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    specs : dict
        Filter specifications
    worN : int
        Number of frequency points
    
    Returns:
    --------
    dict
        Verification results
    """
    fs = specs['fs']
    w, h = signal.freqz(b, a, worN=worN)
    freq = w * fs / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    
    results = {}
    
    # Find magnitude at critical frequencies
    def get_mag_at_freq(f_target):
        idx = np.argmin(np.abs(freq - f_target))
        return mag_db[idx]
    
    # Stopband attenuation at f1
    mag_f1 = get_mag_at_freq(specs['f1'])
    results['mag_at_f1'] = mag_f1
    results['f1_meets_spec'] = mag_f1 <= -specs['k2']
    
    # Passband at fl
    mag_fl = get_mag_at_freq(specs['fl'])
    results['mag_at_fl'] = mag_fl
    results['fl_meets_spec'] = mag_fl >= -specs['k1']
    
    # Passband at fu
    mag_fu = get_mag_at_freq(specs['fu'])
    results['mag_at_fu'] = mag_fu
    results['fu_meets_spec'] = mag_fu >= -specs['k1']
    
    # Stopband attenuation at f2
    mag_f2 = get_mag_at_freq(specs['f2'])
    results['mag_at_f2'] = mag_f2
    results['f2_meets_spec'] = mag_f2 <= -specs['k2']
    
    # Passband ripple (max deviation in passband)
    passband_idx = (freq >= specs['fl']) & (freq <= specs['fu'])
    passband_max = np.max(mag_db[passband_idx])
    passband_min = np.min(mag_db[passband_idx])
    results['passband_max'] = passband_max
    results['passband_min'] = passband_min
    results['passband_ripple'] = passband_max - passband_min
    
    # Overall pass/fail
    results['all_specs_met'] = all([
        results['f1_meets_spec'],
        results['fl_meets_spec'],
        results['fu_meets_spec'],
        results['f2_meets_spec']
    ])
    
    # Print results
    print("\n" + "=" * 60)
    print("SPECIFICATION VERIFICATION RESULTS")
    print("=" * 60)
    
    print(f"\nStopband at f1 = {specs['f1']} Hz:")
    print(f"  Magnitude: {mag_f1:.2f} dB (required ≤ -{specs['k2']} dB)")
    print(f"  Status: {'✓ PASS' if results['f1_meets_spec'] else '✗ FAIL'}")
    
    print(f"\nPassband at fl = {specs['fl']} Hz:")
    print(f"  Magnitude: {mag_fl:.2f} dB (required ≥ -{specs['k1']} dB)")
    print(f"  Status: {'✓ PASS' if results['fl_meets_spec'] else '✗ FAIL'}")
    
    print(f"\nPassband at fu = {specs['fu']} Hz:")
    print(f"  Magnitude: {mag_fu:.2f} dB (required ≥ -{specs['k1']} dB)")
    print(f"  Status: {'✓ PASS' if results['fu_meets_spec'] else '✗ FAIL'}")
    
    print(f"\nStopband at f2 = {specs['f2']} Hz:")
    print(f"  Magnitude: {mag_f2:.2f} dB (required ≤ -{specs['k2']} dB)")
    print(f"  Status: {'✓ PASS' if results['f2_meets_spec'] else '✗ FAIL'}")
    
    print(f"\nPassband Ripple: {results['passband_ripple']:.4f} dB")
    print(f"  Max: {passband_max:.2f} dB, Min: {passband_min:.2f} dB")
    
    print("\n" + "-" * 60)
    if results['all_specs_met']:
        print("OVERALL: ✓ ALL SPECIFICATIONS MET")
    else:
        print("OVERALL: ✗ SOME SPECIFICATIONS NOT MET")
    print("=" * 60)
    
    return results


def plot_all_responses(b: np.ndarray, a: np.ndarray, fs: float,
                       specs: Optional[dict] = None,
                       save_dir: Optional[str] = None) -> None:
    """
    Generate all filter response plots.
    
    Parameters:
    -----------
    b : ndarray
        Numerator coefficients
    a : ndarray
        Denominator coefficients
    fs : float
        Sampling frequency
    specs : dict, optional
        Filter specifications
    save_dir : str, optional
        Directory to save plots
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Magnitude response
    save_path = os.path.join(save_dir, 'magnitude_response.png') if save_dir else None
    plot_magnitude_response(b, a, fs, specs=specs, save_path=save_path)
    
    # Phase response
    save_path = os.path.join(save_dir, 'phase_response.png') if save_dir else None
    plot_phase_response(b, a, fs, save_path=save_path)
    
    # Pole-zero
    save_path = os.path.join(save_dir, 'pole_zero.png') if save_dir else None
    plot_pole_zero(b, a, save_path=save_path)
    
    # Impulse response
    save_path = os.path.join(save_dir, 'impulse_response.png') if save_dir else None
    plot_impulse_response(b, a, save_path=save_path)
    
    # Step response
    save_path = os.path.join(save_dir, 'step_response.png') if save_dir else None
    plot_step_response(b, a, save_path=save_path)
    
    # Group delay
    save_path = os.path.join(save_dir, 'group_delay.png') if save_dir else None
    plot_group_delay(b, a, fs, save_path=save_path)
    
    # Specifications template
    if specs:
        save_path = os.path.join(save_dir, 'specifications.png') if save_dir else None
        plot_filter_specifications(specs, save_path=save_path)


if __name__ == "__main__":
    from filter_design import design_butterworth_bpf
    
    print("Testing Filter Analysis Module")
    print("=" * 60)
    
    # Filter specifications
    specs = {
        'f1': 770,    # Hz (lower stopband)
        'fl': 920,    # Hz (lower cutoff)
        'fu': 1040,   # Hz (upper cutoff)
        'f2': 1155,   # Hz (upper stopband)
        'k1': 2,      # dB (passband ripple)
        'k2': 40,     # dB (stopband attenuation)
        'fs': 6000    # Hz (sampling)
    }
    
    # Design filter
    order = 4
    b, a = design_butterworth_bpf(order, specs['fl'], specs['fu'], specs['fs'])
    
    # Verify specifications
    results = verify_specifications(b, a, specs)
    
    # Plot all responses
    plot_all_responses(b, a, specs['fs'], specs=specs)
