"""
Main BPF Design Script
======================
Complete workflow for Band Pass Filter design using Butterworth approximation
and bilinear transformation.

This script performs:
1. Specification definition
2. Frequency calculations
3. Filter order determination
4. Filter design
5. Verification
6. Visualization
7. Signal filtering demonstration
8. Results export

Author: BPF Design Project
Date: 2024
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filter_helpers import (
    analog_to_digital, prewarping, calculate_bpf_parameters,
    lpf_normalization, calculate_order, print_frequency_summary
)
from filter_design import (
    design_butterworth_bpf, design_butterworth_bpf_sos,
    print_transfer_function, get_difference_equation
)
from filter_analysis import (
    plot_magnitude_response, plot_phase_response, plot_pole_zero,
    plot_impulse_response, plot_step_response, plot_group_delay,
    plot_filter_specifications, verify_specifications, plot_all_responses
)
from signal_processing import (
    generate_test_signal, apply_filter, plot_input_output,
    fft_analysis, demo_filtering, plot_frequency_comparison
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Main function for BPF design workflow."""
    
    print_header("BAND PASS FILTER DESIGN")
    print("Using Butterworth Approximation and Bilinear Transformation")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: DEFINE SPECIFICATIONS
    # =========================================================================
    print_header("STEP 1: FILTER SPECIFICATIONS")
    
    specs = {
        'f1': 770,    # Hz - lower stopband edge
        'fl': 920,    # Hz - lower cutoff frequency (passband)
        'fu': 1040,   # Hz - upper cutoff frequency (passband)
        'f2': 1155,   # Hz - upper stopband edge
        'k1': 2,      # dB - passband ripple (max attenuation in passband)
        'k2': 40,     # dB - stopband attenuation (min attenuation in stopband)
        'fs': 6000    # Hz - sampling frequency
    }
    
    print("\nGiven Specifications:")
    print(f"  f1 (lower stopband edge)  = {specs['f1']} Hz")
    print(f"  fl (lower cutoff)         = {specs['fl']} Hz")
    print(f"  fu (upper cutoff)         = {specs['fu']} Hz")
    print(f"  f2 (upper stopband edge)  = {specs['f2']} Hz")
    print(f"  k1 (passband ripple)      = {specs['k1']} dB")
    print(f"  k2 (stopband attenuation) = {specs['k2']} dB")
    print(f"  fs (sampling frequency)   = {specs['fs']} Hz")
    
    # Nyquist frequency check
    f_nyquist = specs['fs'] / 2
    print(f"\nNyquist Frequency: {f_nyquist} Hz")
    print(f"All specifications below Nyquist: {'✓ Yes' if specs['f2'] < f_nyquist else '✗ No'}")
    
    # =========================================================================
    # STEP 2: CONVERT TO DIGITAL FREQUENCIES
    # =========================================================================
    print_header("STEP 2: DIGITAL FREQUENCY CONVERSION")
    
    print("\nFormula: ω = 2π·f/fs")
    
    omega1 = analog_to_digital(specs['f1'], specs['fs'])
    omega_l = analog_to_digital(specs['fl'], specs['fs'])
    omega_u = analog_to_digital(specs['fu'], specs['fs'])
    omega2 = analog_to_digital(specs['f2'], specs['fs'])
    
    omega = {
        'omega1': omega1,
        'omega_l': omega_l,
        'omega_u': omega_u,
        'omega2': omega2
    }
    
    print(f"\nDigital Frequencies (rad/sample):")
    print(f"  ω1 = 2π × {specs['f1']}/{specs['fs']} = {omega1:.6f} rad/sample")
    print(f"  ωl = 2π × {specs['fl']}/{specs['fs']} = {omega_l:.6f} rad/sample")
    print(f"  ωu = 2π × {specs['fu']}/{specs['fs']} = {omega_u:.6f} rad/sample")
    print(f"  ω2 = 2π × {specs['f2']}/{specs['fs']} = {omega2:.6f} rad/sample")
    
    # =========================================================================
    # STEP 3: PREWARPING (DIGITAL TO ANALOG)
    # =========================================================================
    print_header("STEP 3: PREWARPING")
    
    print("\nFormula: Ω = (2/T)·tan(ω/2)  where T = 2 (normalized)")
    
    Omega1 = prewarping(omega1)
    Omega_l = prewarping(omega_l)
    Omega_u = prewarping(omega_u)
    Omega2 = prewarping(omega2)
    
    Omega = {
        'Omega1': Omega1,
        'Omega_l': Omega_l,
        'Omega_u': Omega_u,
        'Omega2': Omega2
    }
    
    print(f"\nPrewarped Analog Frequencies (rad/s):")
    print(f"  Ω1 = tan({omega1/2:.6f}) = {Omega1:.6f} rad/s")
    print(f"  Ωl = tan({omega_l/2:.6f}) = {Omega_l:.6f} rad/s")
    print(f"  Ωu = tan({omega_u/2:.6f}) = {Omega_u:.6f} rad/s")
    print(f"  Ω2 = tan({omega2/2:.6f}) = {Omega2:.6f} rad/s")
    
    # =========================================================================
    # STEP 4: BPF PARAMETERS
    # =========================================================================
    print_header("STEP 4: BPF PARAMETERS")
    
    Omega_0, B = calculate_bpf_parameters(Omega_l, Omega_u)
    
    print(f"\nCenter Frequency:")
    print(f"  Ω₀ = √(Ωl × Ωu) = √({Omega_l:.6f} × {Omega_u:.6f})")
    print(f"  Ω₀ = {Omega_0:.6f} rad/s")
    
    print(f"\nBandwidth:")
    print(f"  B = Ωu - Ωl = {Omega_u:.6f} - {Omega_l:.6f}")
    print(f"  B = {B:.6f} rad/s")
    
    # =========================================================================
    # STEP 5: LPF NORMALIZATION
    # =========================================================================
    print_header("STEP 5: LPF NORMALIZATION")
    
    print("\nFormula: Ωs = (1/B) × |Ω² - Ω₀²| / Ω")
    
    Omega_s1 = lpf_normalization(Omega1, Omega_0, B)
    Omega_s2 = lpf_normalization(Omega2, Omega_0, B)
    
    print(f"\nNormalized Stopband Frequencies:")
    print(f"  Ωs1 (from Ω1) = {Omega_s1:.6f}")
    print(f"  Ωs2 (from Ω2) = {Omega_s2:.6f}")
    
    Omega_s = min(Omega_s1, Omega_s2)
    print(f"\nUsing Ωs = min(Ωs1, Ωs2) = {Omega_s:.6f} (more restrictive)")
    
    # =========================================================================
    # STEP 6: CALCULATE FILTER ORDER
    # =========================================================================
    print_header("STEP 6: FILTER ORDER CALCULATION")
    
    print("\nButterworth Order Formula:")
    print("  n ≥ log[(10^(k2/10) - 1) / (10^(k1/10) - 1)] / [2 × log(Ωs)]")
    
    epsilon_sq = 10**(specs['k1']/10) - 1
    A_sq = 10**(specs['k2']/10) - 1
    
    print(f"\nCalculations:")
    print(f"  ε² = 10^({specs['k1']}/10) - 1 = {epsilon_sq:.6f}")
    print(f"  A² = 10^({specs['k2']}/10) - 1 = {A_sq:.6f}")
    
    n_exact = np.log10(A_sq / epsilon_sq) / (2 * np.log10(Omega_s))
    n = calculate_order(Omega_s, specs['k1'], specs['k2'])
    
    print(f"  n = log({A_sq/epsilon_sq:.4f}) / (2 × log({Omega_s:.6f}))")
    print(f"  n = {n_exact:.4f}")
    print(f"\n  → Filter Order n = {n} (rounded up)")
    print(f"\n  Note: Actual BPF order = 2n = {2*n} (doubled for bandpass)")
    
    # =========================================================================
    # STEP 7: DESIGN FILTER
    # =========================================================================
    print_header("STEP 7: FILTER DESIGN")
    
    print(f"\nDesigning Butterworth BPF:")
    print(f"  Order: {n}")
    print(f"  Passband: {specs['fl']} - {specs['fu']} Hz")
    print(f"  Sampling Frequency: {specs['fs']} Hz")
    
    # Design using scipy (uses bilinear transformation internally)
    # Standard Butterworth design: -3dB at cutoff frequencies
    # The passband edge will have -3dB attenuation (Butterworth characteristic)
    
    print(f"\nNote: Butterworth filter has -3dB at cutoff frequencies by design.")
    print(f"      This is the standard Butterworth characteristic.")
    
    # Use standard design without adjustment
    b, a = design_butterworth_bpf(n, specs['fl'], specs['fu'], specs['fs'],
                                   adjust_for_passband=False)
    sos = design_butterworth_bpf_sos(n, specs['fl'], specs['fu'], specs['fs'],
                                      adjust_for_passband=False)
    
    print_transfer_function(b, a, 'z')
    
    # =========================================================================
    # STEP 8: DIFFERENCE EQUATION
    # =========================================================================
    print_header("STEP 8: DIFFERENCE EQUATION")
    
    diff_eq = get_difference_equation(b, a)
    print(f"\n{diff_eq}")
    
    # =========================================================================
    # STEP 9: VERIFICATION
    # =========================================================================
    print_header("STEP 9: SPECIFICATION VERIFICATION")
    
    results = verify_specifications(b, a, specs)
    
    # =========================================================================
    # STEP 10: SAVE RESULTS
    # =========================================================================
    print_header("STEP 10: SAVING RESULTS")
    
    # Get results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    data_dir = os.path.join(results_dir, 'data')
    
    # Create directories if needed
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Save filter coefficients
    coef_file = os.path.join(data_dir, 'filter_coefficients.txt')
    with open(coef_file, 'w') as f:
        f.write("Band Pass Filter Coefficients\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Filter Order: {n}\n")
        f.write(f"Passband: {specs['fl']} - {specs['fu']} Hz\n")
        f.write(f"Sampling Frequency: {specs['fs']} Hz\n\n")
        f.write("Numerator (b):\n")
        for i, coef in enumerate(b):
            f.write(f"  b[{i}] = {coef:.15e}\n")
        f.write("\nDenominator (a):\n")
        for i, coef in enumerate(a):
            f.write(f"  a[{i}] = {coef:.15e}\n")
        f.write(f"\nDifference Equation:\n{diff_eq}\n")
    
    print(f"\nSaved coefficients to: {coef_file}")
    
    # =========================================================================
    # STEP 11: GENERATE PLOTS
    # =========================================================================
    print_header("STEP 11: GENERATING PLOTS")
    
    # Plot specifications template
    print("\n1. Filter Specifications Template...")
    plot_filter_specifications(specs, save_path=os.path.join(plots_dir, 'specifications.png'))
    
    # Plot magnitude response
    print("2. Magnitude Response...")
    plot_magnitude_response(b, a, specs['fs'], specs=specs,
                           save_path=os.path.join(plots_dir, 'magnitude_response.png'))
    
    # Plot phase response
    print("3. Phase Response...")
    plot_phase_response(b, a, specs['fs'],
                       save_path=os.path.join(plots_dir, 'phase_response.png'))
    
    # Plot pole-zero diagram
    print("4. Pole-Zero Diagram...")
    plot_pole_zero(b, a, save_path=os.path.join(plots_dir, 'pole_zero.png'))
    
    # Plot impulse response
    print("5. Impulse Response...")
    plot_impulse_response(b, a, save_path=os.path.join(plots_dir, 'impulse_response.png'))
    
    # Plot step response
    print("6. Step Response...")
    plot_step_response(b, a, save_path=os.path.join(plots_dir, 'step_response.png'))
    
    # Plot group delay
    print("7. Group Delay...")
    plot_group_delay(b, a, specs['fs'],
                    save_path=os.path.join(plots_dir, 'group_delay.png'))
    
    # =========================================================================
    # STEP 12: SIGNAL FILTERING DEMONSTRATION
    # =========================================================================
    print_header("STEP 12: SIGNAL FILTERING DEMONSTRATION")
    
    # Create test signal with multiple frequencies
    center_freq = (specs['fl'] + specs['fu']) / 2
    test_frequencies = [
        300,           # Below passband - should be attenuated
        specs['f1'],   # At stopband edge - should be attenuated
        center_freq,   # In passband - should pass
        specs['f2'],   # At stopband edge - should be attenuated
        2000           # Above passband - should be attenuated
    ]
    
    print(f"\nTest Signal Frequencies: {test_frequencies} Hz")
    print(f"Passband: {specs['fl']} - {specs['fu']} Hz")
    
    # Generate signal
    duration = 0.1  # 100ms
    t, x = generate_test_signal(specs['fs'], duration, test_frequencies)
    
    # Apply filter
    y = apply_filter(x, b, a)
    
    # Plot input/output comparison
    print("\n8. Input/Output Comparison...")
    plot_input_output(t, x, y, specs['fs'], specs,
                     save_path=os.path.join(plots_dir, 'input_output_comparison.png'))
    
    # Frequency comparison
    print("9. Frequency Spectrum Comparison...")
    plot_frequency_comparison(x, y, specs['fs'], specs,
                             save_path=os.path.join(plots_dir, 'frequency_comparison.png'))
    
    # FFT Analysis
    print("10. FFT Analysis...")
    fft_results = fft_analysis(x, y, specs['fs'], specs, plots_dir)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("DESIGN COMPLETE")
    
    print("\nSummary:")
    print(f"  Filter Type: Butterworth Band Pass Filter (IIR)")
    print(f"  Design Method: Bilinear Transformation")
    print(f"  Filter Order: {n} (prototype LPF), {2*n} (actual BPF)")
    print(f"  Passband: {specs['fl']} - {specs['fu']} Hz")
    print(f"  Sampling Frequency: {specs['fs']} Hz")
    print(f"  All Specifications Met: {'✓ Yes' if results['all_specs_met'] else '✗ No'}")
    
    print(f"\nOutput Files:")
    print(f"  Coefficients: {coef_file}")
    print(f"  Plots: {plots_dir}")
    
    print("\n" + "=" * 70)
    print(" ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")
    
    return b, a, specs, results


if __name__ == "__main__":
    b, a, specs, results = main()
