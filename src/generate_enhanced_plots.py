"""
=====================================================================
GENERATE ENHANCED PLOTS UNTUK LAPORAN BPF
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os

# Setup
os.makedirs('../results/plots', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# Spesifikasi
specs = {
    'f1': 770, 'fl': 920, 'fu': 1040, 'f2': 1155,
    'k1': 2, 'k2': 40, 'fs': 6000
}
n = 5

# Design filter
Wn = [specs['fl'] / (specs['fs']/2), specs['fu'] / (specs['fs']/2)]
b, a = signal.butter(n, Wn, btype='band')

# Frequency response
w, h = signal.freqz(b, a, worN=8192)
freq = w * specs['fs'] / (2 * np.pi)
mag_db = 20 * np.log10(np.abs(h) + 1e-12)
phase = np.unwrap(np.angle(h))

print("Generating enhanced plots...")

# =====================================================================
# PLOT 1: SPECIFICATION TEMPLATE OVERLAY
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Plot actual response
ax.plot(freq, mag_db, 'b-', linewidth=2, label='Actual Response')

# Specification template
# Passband
ax.fill_between([specs['fl'], specs['fu']], -specs['k1'], 5, 
                alpha=0.2, color='green', label='Passband Spec')
ax.axhline(-specs['k1'], color='green', linestyle='--', linewidth=1.5)

# Stopband regions
ax.fill_between([0, specs['f1']], -100, -specs['k2'], 
                alpha=0.2, color='red', label='Stopband Spec')
ax.fill_between([specs['f2'], specs['fs']/2], -100, -specs['k2'], 
                alpha=0.2, color='red')
ax.axhline(-specs['k2'], color='red', linestyle='--', linewidth=1.5)

# Mark critical frequencies
for f, name, color in [(specs['f1'], 'f₁=770Hz', 'red'),
                        (specs['fl'], 'fₗ=920Hz', 'orange'),
                        (specs['fu'], 'fᵤ=1040Hz', 'orange'),
                        (specs['f2'], 'f₂=1155Hz', 'red')]:
    ax.axvline(f, color=color, linestyle=':', alpha=0.7)
    ax.annotate(name, (f, -5), rotation=90, fontsize=9, 
                ha='right', va='bottom')

# Annotations for measured values
def get_mag(f):
    idx = np.argmin(np.abs(freq - f))
    return mag_db[idx]

ax.annotate(f'{get_mag(specs["f1"]):.1f}dB', 
            (specs['f1'], get_mag(specs['f1'])), 
            xytext=(specs['f1']-80, get_mag(specs['f1'])+10),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

ax.annotate(f'{get_mag(specs["f2"]):.1f}dB', 
            (specs['f2'], get_mag(specs['f2'])), 
            xytext=(specs['f2']+50, get_mag(specs['f2'])+10),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Magnitude (dB)', fontsize=12)
ax.set_title('Band Pass Filter - Magnitude Response with Specification Template', fontsize=14)
ax.set_xlim([0, specs['fs']/2])
ax.set_ylim([-80, 5])
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/specification_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ specification_overlay.png")

# =====================================================================
# PLOT 2: PASSBAND DETAIL WITH -3dB MARKING
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(freq, mag_db, 'b-', linewidth=2)

# -3dB line (Butterworth characteristic)
ax.axhline(-3, color='purple', linestyle='--', linewidth=1.5, label='-3dB (Butterworth cutoff)')
ax.axhline(-specs['k1'], color='green', linestyle='--', linewidth=1.5, label=f'-{specs["k1"]}dB (Spec)')
ax.axhline(0, color='black', linestyle='-', alpha=0.3)

# Mark cutoff frequencies
ax.axvline(specs['fl'], color='orange', linestyle='--', label=f'fₗ={specs["fl"]}Hz')
ax.axvline(specs['fu'], color='orange', linestyle='--', label=f'fᵤ={specs["fu"]}Hz')

# Center frequency
fc = (specs['fl'] + specs['fu']) / 2
ax.axvline(fc, color='green', linestyle=':', alpha=0.7)
ax.annotate(f'fc={fc:.0f}Hz\n{get_mag(fc):.2f}dB', (fc, get_mag(fc)+0.5), 
            ha='center', fontsize=10)

ax.set_xlim([specs['fl']-150, specs['fu']+150])
ax.set_ylim([-10, 2])
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Magnitude (dB)', fontsize=12)
ax.set_title('Passband Detail - Butterworth Characteristic (-3dB at Cutoff)', fontsize=14)
ax.legend(loc='lower center')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/passband_detail.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ passband_detail.png")

# =====================================================================
# PLOT 3: BEFORE/AFTER FILTERING (4 SUBPLOTS)
# =====================================================================
# Generate test signal
duration = 0.05
n_samples = int(specs['fs'] * duration)
t = np.arange(n_samples) / specs['fs']

# Multiple frequency components
fc = (specs['fl'] + specs['fu']) / 2
test_freqs = [300, specs['f1'], fc, specs['f2'], 2000]

x = np.zeros(n_samples)
for f in test_freqs:
    x += np.sin(2 * np.pi * f * t)

# Apply filter
y = signal.filtfilt(b, a, x)

# FFT
def compute_fft(sig, fs):
    n = len(sig)
    freq = fftfreq(n, 1/fs)[:n//2]
    mag = 2/n * np.abs(fft(sig)[:n//2])
    return freq, mag

freq_x, mag_x = compute_fft(x, specs['fs'])
freq_y, mag_y = compute_fft(y, specs['fs'])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Input time domain
ax = axes[0, 0]
ax.plot(t*1000, x, 'b-', linewidth=0.8)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude')
ax.set_title('(a) Input Signal x(t) - Time Domain')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 20])

# Input frequency domain
ax = axes[0, 1]
ax.plot(freq_x, mag_x, 'b-', linewidth=1)
ax.axvspan(specs['fl'], specs['fu'], alpha=0.2, color='green', label='Passband')
for f in test_freqs:
    ax.axvline(f, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.set_title('(b) Input Spectrum X(f)')
ax.legend()
ax.set_xlim([0, specs['fs']/2])
ax.grid(True, alpha=0.3)

# Output time domain
ax = axes[1, 0]
ax.plot(t*1000, y, 'r-', linewidth=0.8)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude')
ax.set_title('(c) Output Signal y(t) - Time Domain')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 20])

# Output frequency domain
ax = axes[1, 1]
ax.plot(freq_y, mag_y, 'r-', linewidth=1)
ax.axvspan(specs['fl'], specs['fu'], alpha=0.2, color='green', label='Passband')
for f in test_freqs:
    ax.axvline(f, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.set_title('(d) Output Spectrum Y(f)')
ax.legend()
ax.set_xlim([0, specs['fs']/2])
ax.grid(True, alpha=0.3)

plt.suptitle('Before/After Filtering Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/plots/before_after_filtering.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ before_after_filtering.png")

# =====================================================================
# PLOT 4: POLE-ZERO WITH UNIT CIRCLE (ENHANCED)
# =====================================================================
zeros = np.roots(b)
poles = np.roots(a)

fig, ax = plt.subplots(figsize=(10, 10))

# Unit circle
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=2, label='Unit Circle')

# Grid circles
for r in [0.5, 0.8, 0.9, 0.95]:
    ax.plot(r*np.cos(theta), r*np.sin(theta), 'gray', linestyle=':', alpha=0.3)
    ax.annotate(f'r={r}', (r*0.7, r*0.7), fontsize=8, color='gray')

# Zeros and poles
ax.scatter(zeros.real, zeros.imag, marker='o', s=100, 
           facecolors='none', edgecolors='blue', linewidths=2, label=f'Zeros ({len(zeros)})')
ax.scatter(poles.real, poles.imag, marker='x', s=100, 
           c='red', linewidths=2, label=f'Poles ({len(poles)})')

# Mark max pole
max_pole_idx = np.argmax(np.abs(poles))
max_pole = poles[max_pole_idx]
ax.annotate(f'Max |p|={np.abs(max_pole):.4f}', 
            (max_pole.real, max_pole.imag),
            xytext=(max_pole.real+0.2, max_pole.imag+0.2),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlabel('Real Part', fontsize=12)
ax.set_ylabel('Imaginary Part', fontsize=12)
ax.set_title('Pole-Zero Plot with Unit Circle\n(All poles inside → STABLE)', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])

# Stability note
ax.text(0.02, 0.98, f'Max |pole| = {np.max(np.abs(poles)):.6f} < 1\n→ STABLE', 
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('../results/plots/pole_zero_enhanced.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ pole_zero_enhanced.png")

# =====================================================================
# PLOT 5: GROUP DELAY
# =====================================================================
w_gd, gd = signal.group_delay((b, a), w=8192)
freq_gd = w_gd * specs['fs'] / (2 * np.pi)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(freq_gd, gd, 'g-', linewidth=1.5)
ax.axvline(specs['fl'], color='orange', linestyle='--', alpha=0.7)
ax.axvline(specs['fu'], color='orange', linestyle='--', alpha=0.7)
ax.axvspan(specs['fl'], specs['fu'], alpha=0.1, color='green')
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Group Delay (samples)', fontsize=12)
ax.set_title('Group Delay Response', fontsize=14)
ax.set_xlim([0, specs['fs']/2])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/group_delay_enhanced.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ group_delay_enhanced.png")

# =====================================================================
# PLOT 6: COMPLETE SUMMARY (6 SUBPLOTS)
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Magnitude
ax = axes[0, 0]
ax.plot(freq, mag_db, 'b-', linewidth=1.5)
ax.axhline(-specs['k1'], color='green', linestyle='--', alpha=0.7)
ax.axhline(-specs['k2'], color='red', linestyle='--', alpha=0.7)
ax.axvspan(specs['fl'], specs['fu'], alpha=0.1, color='green')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude (dB)')
ax.set_title('(a) Magnitude Response')
ax.set_xlim([0, specs['fs']/2])
ax.set_ylim([-80, 5])
ax.grid(True, alpha=0.3)

# Phase
ax = axes[0, 1]
ax.plot(freq, np.degrees(phase), 'r-', linewidth=1.5)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Phase (degrees)')
ax.set_title('(b) Phase Response')
ax.set_xlim([0, specs['fs']/2])
ax.grid(True, alpha=0.3)

# Pole-Zero (mini)
ax = axes[0, 2]
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)
ax.scatter(zeros.real, zeros.imag, marker='o', s=50, 
           facecolors='none', edgecolors='blue', linewidths=1.5)
ax.scatter(poles.real, poles.imag, marker='x', s=50, c='red', linewidths=1.5)
ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.set_title('(c) Pole-Zero Plot')
ax.set_aspect('equal')
ax.set_xlim([-1.3, 1.3])
ax.set_ylim([-1.3, 1.3])
ax.grid(True, alpha=0.3)

# Impulse response
ax = axes[1, 0]
impulse = np.zeros(100)
impulse[0] = 1
h_n = signal.lfilter(b, a, impulse)
ax.stem(range(100), h_n, linefmt='b-', markerfmt='bo', basefmt='k-')
ax.set_xlabel('Sample n')
ax.set_ylabel('Amplitude')
ax.set_title('(d) Impulse Response h[n]')
ax.grid(True, alpha=0.3)

# Step response
ax = axes[1, 1]
step = np.ones(100)
s_n = signal.lfilter(b, a, step)
ax.plot(range(100), s_n, 'g-', linewidth=1.5)
ax.set_xlabel('Sample n')
ax.set_ylabel('Amplitude')
ax.set_title('(e) Step Response')
ax.grid(True, alpha=0.3)

# Group delay
ax = axes[1, 2]
ax.plot(freq_gd, gd, 'purple', linewidth=1.5)
ax.axvspan(specs['fl'], specs['fu'], alpha=0.1, color='green')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Samples')
ax.set_title('(f) Group Delay')
ax.set_xlim([0, specs['fs']/2])
ax.grid(True, alpha=0.3)

plt.suptitle('BPF Design Summary - Butterworth Order 5 (BPF Order 10)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/plots/complete_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ complete_summary.png")

# =====================================================================
# PLOT 7: VERIFICATION TABLE AS IMAGE
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Table data
table_data = [
    ['Parameter', 'Value', 'Formula'],
    ['ω₁', '0.806342 rad/s', '2π×770/6000'],
    ['ωₗ', '0.963422 rad/s', '2π×920/6000'],
    ['ωᵤ', '1.089085 rad/s', '2π×1040/6000'],
    ['ω₂', '1.209513 rad/s', '2π×1155/6000'],
    ['Ω₁', '0.426536 rad/s', 'tan(ω₁/2)'],
    ['Ωₗ', '0.522787 rad/s', 'tan(ωₗ/2)'],
    ['Ωᵤ', '0.605622 rad/s', 'tan(ωᵤ/2)'],
    ['Ω₂', '0.691143 rad/s', 'tan(ω₂/2)'],
    ['Ω₀', '0.562682 rad/s', '√(Ωₗ×Ωᵤ)'],
    ['B', '0.082834 rad/s', 'Ωᵤ - Ωₗ'],
    ['Ωᵣ', '2.813382', 'min(A, B_norm)'],
    ['n', '5', '⌈4.711⌉'],
    ['BPF Order', '10', '2n'],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                 colWidths=[0.2, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Header styling
for i in range(3):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

ax.set_title('Calculation Summary Table', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../results/plots/calculation_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ calculation_table.png")

print("\n" + "="*50)
print("All enhanced plots generated successfully!")
print("="*50)
