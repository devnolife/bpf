"""
Test Filter Module
==================
Unit tests for Band Pass Filter design functions.
"""

import numpy as np
import sys
import os
import unittest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from filter_helpers import (
    analog_to_digital, prewarping, calculate_bpf_parameters,
    lpf_normalization, calculate_order
)
from filter_design import (
    design_butterworth_bpf, design_butterworth_bpf_sos
)


class TestFilterHelpers(unittest.TestCase):
    """Test cases for filter helper functions."""
    
    def test_analog_to_digital(self):
        """Test analog to digital frequency conversion."""
        # At Nyquist frequency, omega should be pi
        fs = 1000
        f = fs / 2  # Nyquist
        omega = analog_to_digital(f, fs)
        self.assertAlmostEqual(omega, np.pi, places=5)
        
        # At fs/4, omega should be pi/2
        omega = analog_to_digital(fs/4, fs)
        self.assertAlmostEqual(omega, np.pi/2, places=5)
    
    def test_prewarping(self):
        """Test prewarping function."""
        # At omega = 0, Omega should be 0
        omega = 0.001  # Small value (avoid exactly 0)
        Omega = prewarping(omega)
        self.assertAlmostEqual(Omega, 0, places=3)
        
        # At omega = pi/2, Omega should be 1 (for T=2)
        omega = np.pi / 2
        Omega = prewarping(omega, T=2)
        self.assertAlmostEqual(Omega, 1.0, places=5)
    
    def test_calculate_bpf_parameters(self):
        """Test BPF parameter calculation."""
        Omega_l = 1.0
        Omega_u = 4.0
        
        Omega_0, B = calculate_bpf_parameters(Omega_l, Omega_u)
        
        # Center frequency should be geometric mean
        self.assertAlmostEqual(Omega_0, 2.0, places=5)  # sqrt(1*4) = 2
        
        # Bandwidth should be difference
        self.assertAlmostEqual(B, 3.0, places=5)  # 4 - 1 = 3
    
    def test_calculate_order(self):
        """Test filter order calculation."""
        # Known case: large Omega_s should give low order
        Omega_s = 10.0
        k1 = 3  # dB
        k2 = 40  # dB
        
        n = calculate_order(Omega_s, k1, k2)
        
        # Order should be positive integer
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)
        
        # For high Omega_s, order should be low
        self.assertLess(n, 10)


class TestFilterDesign(unittest.TestCase):
    """Test cases for filter design functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.specs = {
            'fl': 920,
            'fu': 1040,
            'fs': 6000
        }
        self.order = 4
    
    def test_design_butterworth_bpf_coefficients(self):
        """Test that BPF design returns valid coefficients."""
        b, a = design_butterworth_bpf(
            self.order, 
            self.specs['fl'], 
            self.specs['fu'], 
            self.specs['fs']
        )
        
        # Check coefficient shapes
        # For BPF, order doubles
        expected_len = 2 * self.order + 1
        self.assertEqual(len(b), expected_len)
        self.assertEqual(len(a), expected_len)
        
        # First denominator coefficient should be 1 (normalized)
        self.assertAlmostEqual(a[0], 1.0, places=10)
    
    def test_design_butterworth_bpf_stability(self):
        """Test that designed filter is stable."""
        b, a = design_butterworth_bpf(
            self.order,
            self.specs['fl'],
            self.specs['fu'],
            self.specs['fs']
        )
        
        # Check poles are inside unit circle
        poles = np.roots(a)
        max_pole_mag = np.max(np.abs(poles))
        
        self.assertLess(max_pole_mag, 1.0, 
                       f"Filter unstable: max pole magnitude = {max_pole_mag}")
    
    def test_design_butterworth_bpf_sos(self):
        """Test SOS representation."""
        sos = design_butterworth_bpf_sos(
            self.order,
            self.specs['fl'],
            self.specs['fu'],
            self.specs['fs']
        )
        
        # SOS should have shape (n_sections, 6)
        self.assertEqual(sos.shape[1], 6)
        
        # For BPF order n, we get 2n poles, so n sections
        expected_sections = self.order
        self.assertEqual(sos.shape[0], expected_sections)
    
    def test_passband_response(self):
        """Test that passband has correct gain."""
        from scipy import signal
        
        b, a = design_butterworth_bpf(
            self.order,
            self.specs['fl'],
            self.specs['fu'],
            self.specs['fs']
        )
        
        # Calculate frequency response
        w, h = signal.freqz(b, a, worN=8192)
        freq = w * self.specs['fs'] / (2 * np.pi)
        mag_db = 20 * np.log10(np.abs(h) + 1e-12)
        
        # Find magnitude at center frequency
        center = (self.specs['fl'] + self.specs['fu']) / 2
        idx = np.argmin(np.abs(freq - center))
        center_mag = mag_db[idx]
        
        # Center should be near 0 dB (within 3 dB)
        self.assertGreater(center_mag, -3.0,
                          f"Center frequency gain too low: {center_mag} dB")


class TestFrequencyConversion(unittest.TestCase):
    """Test frequency conversion chain."""
    
    def test_conversion_roundtrip(self):
        """Test that frequency conversions are consistent."""
        fs = 6000
        f_test = 1000  # Hz
        
        # Convert to digital
        omega = analog_to_digital(f_test, fs)
        
        # Prewarp
        Omega = prewarping(omega)
        
        # Check omega is in valid range
        self.assertGreater(omega, 0)
        self.assertLess(omega, np.pi)
        
        # Check Omega is positive
        self.assertGreater(Omega, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete filter design."""
    
    def test_complete_design_workflow(self):
        """Test complete filter design from specs to coefficients."""
        from scipy import signal
        
        # Specifications
        specs = {
            'f1': 770,
            'fl': 920,
            'fu': 1040,
            'f2': 1155,
            'k1': 2,
            'k2': 40,
            'fs': 6000
        }
        
        # Calculate frequencies
        omega_l = analog_to_digital(specs['fl'], specs['fs'])
        omega_u = analog_to_digital(specs['fu'], specs['fs'])
        
        Omega_l = prewarping(omega_l)
        Omega_u = prewarping(omega_u)
        
        Omega_0, B = calculate_bpf_parameters(Omega_l, Omega_u)
        
        omega1 = analog_to_digital(specs['f1'], specs['fs'])
        omega2 = analog_to_digital(specs['f2'], specs['fs'])
        Omega1 = prewarping(omega1)
        Omega2 = prewarping(omega2)
        
        Omega_s1 = lpf_normalization(Omega1, Omega_0, B)
        Omega_s2 = lpf_normalization(Omega2, Omega_0, B)
        Omega_s = min(Omega_s1, Omega_s2)
        
        # Calculate order
        n = calculate_order(Omega_s, specs['k1'], specs['k2'])
        
        # Design filter
        b, a = design_butterworth_bpf(n, specs['fl'], specs['fu'], specs['fs'])
        
        # Verify frequency response
        w, h = signal.freqz(b, a, worN=8192)
        freq = w * specs['fs'] / (2 * np.pi)
        mag_db = 20 * np.log10(np.abs(h) + 1e-12)
        
        def get_mag_at_freq(f_target):
            idx = np.argmin(np.abs(freq - f_target))
            return mag_db[idx]
        
        # Check specifications
        mag_f1 = get_mag_at_freq(specs['f1'])
        mag_fl = get_mag_at_freq(specs['fl'])
        mag_fu = get_mag_at_freq(specs['fu'])
        mag_f2 = get_mag_at_freq(specs['f2'])
        
        # Stopband should be below -k2 dB
        self.assertLess(mag_f1, -specs['k2'],
                       f"Stopband at f1 not met: {mag_f1} dB")
        
        # Passband should be above -k1 dB
        self.assertGreater(mag_fl, -specs['k1'],
                          f"Passband at fl not met: {mag_fl} dB")
        self.assertGreater(mag_fu, -specs['k1'],
                          f"Passband at fu not met: {mag_fu} dB")


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
