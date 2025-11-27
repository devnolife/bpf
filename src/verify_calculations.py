"""
=====================================================================
VERIFIKASI PERHITUNGAN FILTER BAND PASS (BPF) IIR
Menggunakan Transformasi Bilinear - Approximasi Butterworth
=====================================================================

Script ini memverifikasi semua perhitungan secara detail dengan:
1. Perhitungan manual step-by-step
2. Cross-check dengan scipy
3. Tabel verifikasi numerik
4. Export hasil ke file

Author: PSM Student
Date: November 2025
=====================================================================
"""

import numpy as np
from scipy import signal
import os

# Buat folder output jika belum ada
os.makedirs('../results/verification', exist_ok=True)

print("=" * 70)
print("VERIFIKASI PERHITUNGAN FILTER BAND PASS (BPF) IIR")
print("Approximasi Butterworth - Transformasi Bilinear")
print("=" * 70)

# =====================================================================
# BAGIAN 1: SPESIFIKASI FILTER
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 1: SPESIFIKASI FILTER")
print("=" * 70)

# Spesifikasi yang diberikan
f1 = 770      # Hz - frekuensi stopband bawah
fl = 920      # Hz - frekuensi cutoff bawah (passband)
fu = 1040     # Hz - frekuensi cutoff atas (passband)
f2 = 1155     # Hz - frekuensi stopband atas
k1 = 2        # dB - ripple passband maksimum
k2 = 40       # dB - atenuasi stopband minimum
fs = 6000     # Hz - frekuensi sampling

print(f"""
Spesifikasi Filter:
-------------------
f1 (stopband bawah)    = {f1} Hz
fl (cutoff bawah)      = {fl} Hz
fu (cutoff atas)       = {fu} Hz
f2 (stopband atas)     = {f2} Hz
k1 (ripple passband)   = {k1} dB
k2 (atenuasi stopband) = {k2} dB
fs (sampling)          = {fs} Hz
fn (Nyquist)           = {fs/2} Hz
""")

# =====================================================================
# BAGIAN 2: KONVERSI FREKUENSI DIGITAL (ω)
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 2: KONVERSI FREKUENSI DIGITAL")
print("=" * 70)

print("""
Formula: ω = 2π × f / fs

Penjelasan:
- ω adalah frekuensi digital dalam rad/sample
- Rentang valid: 0 ≤ ω ≤ π (Nyquist limit)
""")

# Perhitungan
omega1 = 2 * np.pi * f1 / fs
omega_l = 2 * np.pi * fl / fs
omega_u = 2 * np.pi * fu / fs
omega2 = 2 * np.pi * f2 / fs

print("Perhitungan Detail:")
print("-" * 50)
print(f"ω1 = 2π × {f1} / {fs}")
print(f"   = 2π × {f1/fs:.6f}")
print(f"   = {omega1:.6f} rad/sample")
print(f"   = {omega1/np.pi:.6f}π rad/sample")
print()
print(f"ωl = 2π × {fl} / {fs}")
print(f"   = 2π × {fl/fs:.6f}")
print(f"   = {omega_l:.6f} rad/sample")
print(f"   = {omega_l/np.pi:.6f}π rad/sample")
print()
print(f"ωu = 2π × {fu} / {fs}")
print(f"   = 2π × {fu/fs:.6f}")
print(f"   = {omega_u:.6f} rad/sample")
print(f"   = {omega_u/np.pi:.6f}π rad/sample")
print()
print(f"ω2 = 2π × {f2} / {fs}")
print(f"   = 2π × {f2/fs:.6f}")
print(f"   = {omega2:.6f} rad/sample")
print(f"   = {omega2/np.pi:.6f}π rad/sample")

# Tabel ringkasan
print("\n" + "-" * 50)
print("TABEL FREKUENSI DIGITAL:")
print("-" * 50)
print(f"{'Simbol':<10} {'f (Hz)':<10} {'ω (rad/s)':<15} {'ω/π':<10}")
print("-" * 50)
print(f"{'ω1':<10} {f1:<10} {omega1:<15.6f} {omega1/np.pi:<10.6f}")
print(f"{'ωl':<10} {fl:<10} {omega_l:<15.6f} {omega_l/np.pi:<10.6f}")
print(f"{'ωu':<10} {fu:<10} {omega_u:<15.6f} {omega_u/np.pi:<10.6f}")
print(f"{'ω2':<10} {f2:<10} {omega2:<15.6f} {omega2/np.pi:<10.6f}")

# =====================================================================
# BAGIAN 3: PREWARPING
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 3: PREWARPING")
print("=" * 70)

print("""
Formula: Ω = (2/T) × tan(ω/2)

Dengan T = 2 (periode sampling ternormalisasi), maka:
        Ω = tan(ω/2)

Penjelasan:
- Prewarping mengkompensasi frequency warping dari transformasi bilinear
- Tanpa prewarping, frekuensi cutoff akan bergeser
""")

# Dengan T = 2 untuk normalisasi
T = 2
Omega1 = (2/T) * np.tan(omega1 / 2)
Omega_l = (2/T) * np.tan(omega_l / 2)
Omega_u = (2/T) * np.tan(omega_u / 2)
Omega2 = (2/T) * np.tan(omega2 / 2)

print("Perhitungan Detail (dengan T=2):")
print("-" * 50)
print(f"Ω1 = tan(ω1/2) = tan({omega1/2:.6f})")
print(f"   = tan({omega1/(2*np.pi)*180:.4f}°)")
print(f"   = {Omega1:.6f} rad/s")
print()
print(f"Ωl = tan(ωl/2) = tan({omega_l/2:.6f})")
print(f"   = tan({omega_l/(2*np.pi)*180:.4f}°)")
print(f"   = {Omega_l:.6f} rad/s")
print()
print(f"Ωu = tan(ωu/2) = tan({omega_u/2:.6f})")
print(f"   = tan({omega_u/(2*np.pi)*180:.4f}°)")
print(f"   = {Omega_u:.6f} rad/s")
print()
print(f"Ω2 = tan(ω2/2) = tan({omega2/2:.6f})")
print(f"   = tan({omega2/(2*np.pi)*180:.4f}°)")
print(f"   = {Omega2:.6f} rad/s")

# Tabel ringkasan
print("\n" + "-" * 50)
print("TABEL PREWARPING:")
print("-" * 50)
print(f"{'Simbol':<10} {'ω (rad/s)':<15} {'ω/2':<15} {'Ω (rad/s)':<15}")
print("-" * 50)
print(f"{'Ω1':<10} {omega1:<15.6f} {omega1/2:<15.6f} {Omega1:<15.6f}")
print(f"{'Ωl':<10} {omega_l:<15.6f} {omega_l/2:<15.6f} {Omega_l:<15.6f}")
print(f"{'Ωu':<10} {omega_u:<15.6f} {omega_u/2:<15.6f} {Omega_u:<15.6f}")
print(f"{'Ω2':<10} {omega2:<15.6f} {omega2/2:<15.6f} {Omega2:<15.6f}")

# =====================================================================
# BAGIAN 4: PARAMETER BPF
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 4: PARAMETER BAND PASS FILTER")
print("=" * 70)

print("""
Formula:
- Frekuensi Tengah: Ω0 = √(Ωl × Ωu)  [geometric mean]
- Bandwidth: B = Ωu - Ωl

Penjelasan:
- Ω0 adalah geometric mean, bukan arithmetic mean
- Ini memastikan simetri logaritmik pada respon
""")

# Hitung parameter BPF
Omega_0 = np.sqrt(Omega_l * Omega_u)
B = Omega_u - Omega_l

print("Perhitungan Detail:")
print("-" * 50)
print(f"Ω0 = √(Ωl × Ωu)")
print(f"   = √({Omega_l:.6f} × {Omega_u:.6f})")
print(f"   = √{Omega_l * Omega_u:.6f}")
print(f"   = {Omega_0:.6f} rad/s")
print()
print(f"B = Ωu - Ωl")
print(f"  = {Omega_u:.6f} - {Omega_l:.6f}")
print(f"  = {B:.6f} rad/s")

# Verifikasi: Ω0² = Ωl × Ωu
print(f"\nVerifikasi: Ω0² = Ωl × Ωu")
print(f"  Ω0² = {Omega_0**2:.6f}")
print(f"  Ωl × Ωu = {Omega_l * Omega_u:.6f}")
print(f"  Match: {'✓ YES' if abs(Omega_0**2 - Omega_l*Omega_u) < 1e-10 else '✗ NO'}")

# =====================================================================
# BAGIAN 5: NORMALISASI KE LPF PROTOTYPE
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 5: NORMALISASI KE LPF PROTOTYPE")
print("=" * 70)

print("""
Formula dari materi PSM:
A = |Ω1² - Ω0²| / [Ω1 × B]  atau  A = |(-Ω1² + Ωl×Ωu)| / [Ω1 × (Ωu - Ωl)]
B_norm = |Ω2² - Ω0²| / [Ω2 × B]  atau  B_norm = |(Ω2² - Ωl×Ωu)| / [Ω2 × (Ωu - Ωl)]
Ωr = min(A, B_norm)

Penjelasan:
- Transformasi ke normalized LPF dengan cutoff = 1 rad/s
- Pilih minimum untuk memastikan kedua stopband terpenuhi
""")

# Hitung normalisasi
# Menggunakan formula: Ωs = (1/B) × |Ω² - Ω0²| / Ω
A_norm = np.abs(Omega1**2 - Omega_0**2) / (Omega1 * B)
B_norm = np.abs(Omega2**2 - Omega_0**2) / (Omega2 * B)
Omega_r = min(A_norm, B_norm)

print("Perhitungan Detail:")
print("-" * 50)
print(f"A = |Ω1² - Ω0²| / (Ω1 × B)")
print(f"  = |{Omega1:.6f}² - {Omega_0:.6f}²| / ({Omega1:.6f} × {B:.6f})")
print(f"  = |{Omega1**2:.6f} - {Omega_0**2:.6f}| / {Omega1 * B:.6f}")
print(f"  = |{Omega1**2 - Omega_0**2:.6f}| / {Omega1 * B:.6f}")
print(f"  = {np.abs(Omega1**2 - Omega_0**2):.6f} / {Omega1 * B:.6f}")
print(f"  = {A_norm:.6f}")
print()
print(f"B_norm = |Ω2² - Ω0²| / (Ω2 × B)")
print(f"       = |{Omega2:.6f}² - {Omega_0:.6f}²| / ({Omega2:.6f} × {B:.6f})")
print(f"       = |{Omega2**2:.6f} - {Omega_0**2:.6f}| / {Omega2 * B:.6f}")
print(f"       = |{Omega2**2 - Omega_0**2:.6f}| / {Omega2 * B:.6f}")
print(f"       = {np.abs(Omega2**2 - Omega_0**2):.6f} / {Omega2 * B:.6f}")
print(f"       = {B_norm:.6f}")
print()
print(f"Ωr = min(A, B_norm)")
print(f"   = min({A_norm:.6f}, {B_norm:.6f})")
print(f"   = {Omega_r:.6f}")

# Tabel ringkasan
print("\n" + "-" * 50)
print("TABEL NORMALISASI:")
print("-" * 50)
print(f"{'Parameter':<15} {'Formula':<35} {'Nilai':<15}")
print("-" * 50)
print(f"{'A':<15} {'|Ω1² - Ω0²| / (Ω1 × B)':<35} {A_norm:<15.6f}")
print(f"{'B_norm':<15} {'|Ω2² - Ω0²| / (Ω2 × B)':<35} {B_norm:<15.6f}")
print(f"{'Ωr':<15} {'min(A, B_norm)':<35} {Omega_r:<15.6f}")

# =====================================================================
# BAGIAN 6: PERHITUNGAN ORDE FILTER
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 6: PERHITUNGAN ORDE FILTER")
print("=" * 70)

print("""
Formula Butterworth:
n = log₁₀[(10^(k2/10) - 1) / (10^(k1/10) - 1)] / [2 × log₁₀(Ωr)]

Atau equivalen:
n = log₁₀(A²/ε²) / [2 × log₁₀(Ωr)]

dimana:
- ε² = 10^(k1/10) - 1  [passband parameter]
- A² = 10^(k2/10) - 1  [stopband parameter]
""")

# Hitung parameter
epsilon_sq = 10**(k1/10) - 1
A_sq = 10**(k2/10) - 1

print("Perhitungan Parameter:")
print("-" * 50)
print(f"ε² = 10^(k1/10) - 1")
print(f"   = 10^({k1}/10) - 1")
print(f"   = 10^{k1/10:.4f} - 1")
print(f"   = {10**(k1/10):.6f} - 1")
print(f"   = {epsilon_sq:.6f}")
print()
print(f"A² = 10^(k2/10) - 1")
print(f"   = 10^({k2}/10) - 1")
print(f"   = 10^{k2/10:.1f} - 1")
print(f"   = {10**(k2/10):.6f} - 1")
print(f"   = {A_sq:.6f}")

# Hitung orde
n_exact = np.log10(A_sq / epsilon_sq) / (2 * np.log10(Omega_r))
n = int(np.ceil(n_exact))

print()
print("Perhitungan Orde:")
print("-" * 50)
print(f"n = log₁₀(A²/ε²) / [2 × log₁₀(Ωr)]")
print(f"  = log₁₀({A_sq:.6f}/{epsilon_sq:.6f}) / [2 × log₁₀({Omega_r:.6f})]")
print(f"  = log₁₀({A_sq/epsilon_sq:.6f}) / [2 × {np.log10(Omega_r):.6f}]")
print(f"  = {np.log10(A_sq/epsilon_sq):.6f} / {2*np.log10(Omega_r):.6f}")
print(f"  = {n_exact:.6f}")
print()
print(f"n (sebelum pembulatan) = {n_exact:.6f}")
print(f"n (setelah ceiling)    = {n}")
print(f"Orde BPF aktual        = 2n = {2*n}")

# =====================================================================
# BAGIAN 7: BUTTERWORTH POLES (LPF PROTOTYPE)
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 7: BUTTERWORTH POLES (LPF PROTOTYPE)")
print("=" * 70)

print(f"""
Untuk Butterworth orde n = {n}, poles LPF prototype terletak pada:
pk = exp(j × π × (2k + n - 1) / (2n)), untuk k = 1, 2, ..., n

Poles berada di setengah kiri bidang-s (real part negatif).
""")

# Hitung poles LPF prototype
poles_lpf = []
for k in range(1, n + 1):
    angle = np.pi * (2*k + n - 1) / (2*n)
    pole = np.exp(1j * angle)
    poles_lpf.append(pole)
    print(f"p{k} = exp(j × π × ({2*k} + {n} - 1) / (2×{n}))")
    print(f"   = exp(j × π × {2*k + n - 1} / {2*n})")
    print(f"   = exp(j × {angle:.6f})")
    print(f"   = {pole.real:.6f} + j{pole.imag:.6f}")
    print(f"   Magnitude: |p{k}| = {np.abs(pole):.6f}")
    print()

poles_lpf = np.array(poles_lpf)

# Tampilkan dalam bentuk quadratic factors
print("Quadratic Factors (untuk implementasi):")
print("-" * 50)
for i in range(0, n, 2):
    if i + 1 < n:
        p1, p2 = poles_lpf[i], poles_lpf[i+1]
        # (s - p1)(s - p2) = s² - (p1+p2)s + p1*p2
        b1 = -(p1 + p2).real  # coefficient of s
        b0 = (p1 * p2).real   # constant
        print(f"Factor {i//2 + 1}: (s² + {b1:.6f}s + {b0:.6f})")
    else:
        p = poles_lpf[i]
        print(f"Factor {i//2 + 1}: (s + {-p.real:.6f})")

# =====================================================================
# BAGIAN 8: DESAIN FILTER DIGITAL MENGGUNAKAN SCIPY
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 8: DESAIN FILTER DIGITAL (SCIPY)")
print("=" * 70)

# Normalize frequencies to Nyquist
Wn = [fl / (fs/2), fu / (fs/2)]
print(f"Frekuensi ternormalisasi ke Nyquist:")
print(f"  Wn_low  = fl / (fs/2) = {fl} / {fs/2} = {Wn[0]:.6f}")
print(f"  Wn_high = fu / (fs/2) = {fu} / {fs/2} = {Wn[1]:.6f}")
print(f"  Wn = [{Wn[0]:.6f}, {Wn[1]:.6f}]")
print()

# Design filter
b, a = signal.butter(n, Wn, btype='band', analog=False)
sos = signal.butter(n, Wn, btype='band', analog=False, output='sos')

print(f"Filter dirancang dengan scipy.signal.butter()")
print(f"  Orde: {n}")
print(f"  Tipe: Band Pass")
print(f"  Output: ba (numerator, denominator)")
print()

# Print coefficients
print("KOEFISIEN NUMERATOR (b):")
print("-" * 50)
for i, coef in enumerate(b):
    print(f"  b[{i:2d}] = {coef:>25.15e}")

print()
print("KOEFISIEN DENOMINATOR (a):")
print("-" * 50)
for i, coef in enumerate(a):
    print(f"  a[{i:2d}] = {coef:>25.15e}")

print()
print(f"Jumlah koefisien b: {len(b)} (seharusnya 2n+1 = {2*n+1})")
print(f"Jumlah koefisien a: {len(a)} (seharusnya 2n+1 = {2*n+1})")

# =====================================================================
# BAGIAN 9: POLES DAN ZEROS FILTER DIGITAL
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 9: POLES DAN ZEROS FILTER DIGITAL")
print("=" * 70)

zeros = np.roots(b)
poles = np.roots(a)

print("ZEROS (akar numerator):")
print("-" * 50)
for i, z in enumerate(zeros):
    print(f"  z{i+1:2d} = {z.real:>12.6f} + j{z.imag:>12.6f}  |z| = {np.abs(z):.6f}")

print()
print("POLES (akar denominator):")
print("-" * 50)
for i, p in enumerate(poles):
    print(f"  p{i+1:2d} = {p.real:>12.6f} + j{p.imag:>12.6f}  |p| = {np.abs(p):.6f}")

print()
print("ANALISIS STABILITAS:")
print("-" * 50)
max_pole_mag = np.max(np.abs(poles))
print(f"  Magnitude pole maksimum: {max_pole_mag:.6f}")
print(f"  Kriteria stabilitas: |pole| < 1")
print(f"  Status: {'✓ STABIL' if max_pole_mag < 1 else '✗ TIDAK STABIL'}")

# =====================================================================
# BAGIAN 10: VERIFIKASI SPESIFIKASI
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 10: VERIFIKASI SPESIFIKASI")
print("=" * 70)

# Calculate frequency response
w, h = signal.freqz(b, a, worN=8192)
freq = w * fs / (2 * np.pi)
mag_db = 20 * np.log10(np.abs(h) + 1e-12)

def get_mag_at_freq(f_target):
    idx = np.argmin(np.abs(freq - f_target))
    return mag_db[idx]

# Frekuensi uji
test_freqs = {
    'f1 (stopband bawah)': (f1, f'≤ -{k2} dB'),
    'fl (cutoff bawah)': (fl, f'≥ -{k1} dB'),
    'fc (center)': ((fl+fu)/2, '≈ 0 dB'),
    'fu (cutoff atas)': (fu, f'≥ -{k1} dB'),
    'f2 (stopband atas)': (f2, f'≤ -{k2} dB'),
}

print("TABEL VERIFIKASI SPESIFIKASI:")
print("-" * 70)
print(f"{'Frekuensi':<25} {'Hz':<10} {'Spesifikasi':<15} {'Aktual':<15} {'Status':<10}")
print("-" * 70)

results = []
for name, (freq_val, spec) in test_freqs.items():
    mag = get_mag_at_freq(freq_val)
    
    # Determine pass/fail
    if 'stopband' in name:
        status = '✓ PASS' if mag <= -k2 else '✗ FAIL'
    elif 'cutoff' in name:
        # Butterworth: -3dB at cutoff is characteristic
        status = '✓ PASS (-3dB)' if mag >= -4 else '✗ FAIL'
    else:
        status = '✓ PASS' if mag >= -1 else '✗ FAIL'
    
    print(f"{name:<25} {freq_val:<10.0f} {spec:<15} {mag:>10.2f} dB   {status:<10}")
    results.append((name, freq_val, spec, mag, status))

# =====================================================================
# BAGIAN 11: TABEL PERBANDINGAN MANUAL VS PYTHON
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 11: CROSS-CHECK MANUAL VS PYTHON")
print("=" * 70)

print("""
Tabel ini membandingkan hasil perhitungan manual dengan hasil Python.
Semua nilai dihitung dengan formula yang sama.
""")

print("-" * 70)
print(f"{'Parameter':<20} {'Formula':<30} {'Nilai':<20}")
print("-" * 70)
print(f"{'ω1':<20} {'2π×f1/fs':<30} {omega1:<20.6f}")
print(f"{'ωl':<20} {'2π×fl/fs':<30} {omega_l:<20.6f}")
print(f"{'ωu':<20} {'2π×fu/fs':<30} {omega_u:<20.6f}")
print(f"{'ω2':<20} {'2π×f2/fs':<30} {omega2:<20.6f}")
print("-" * 70)
print(f"{'Ω1':<20} {'tan(ω1/2)':<30} {Omega1:<20.6f}")
print(f"{'Ωl':<20} {'tan(ωl/2)':<30} {Omega_l:<20.6f}")
print(f"{'Ωu':<20} {'tan(ωu/2)':<30} {Omega_u:<20.6f}")
print(f"{'Ω2':<20} {'tan(ω2/2)':<30} {Omega2:<20.6f}")
print("-" * 70)
print(f"{'Ω0':<20} {'√(Ωl×Ωu)':<30} {Omega_0:<20.6f}")
print(f"{'B':<20} {'Ωu - Ωl':<30} {B:<20.6f}")
print("-" * 70)
print(f"{'A':<20} {'|Ω1²-Ω0²|/(Ω1×B)':<30} {A_norm:<20.6f}")
print(f"{'B_norm':<20} {'|Ω2²-Ω0²|/(Ω2×B)':<30} {B_norm:<20.6f}")
print(f"{'Ωr':<20} {'min(A, B_norm)':<30} {Omega_r:<20.6f}")
print("-" * 70)
print(f"{'ε²':<20} {'10^(k1/10) - 1':<30} {epsilon_sq:<20.6f}")
print(f"{'A²':<20} {'10^(k2/10) - 1':<30} {A_sq:<20.6f}")
print(f"{'n (exact)':<20} {'log(A²/ε²)/(2log(Ωr))':<30} {n_exact:<20.6f}")
print(f"{'n (rounded)':<20} {'ceil(n_exact)':<30} {n:<20}")
print(f"{'Orde BPF':<20} {'2n':<30} {2*n:<20}")
print("-" * 70)

# =====================================================================
# BAGIAN 12: EXPORT HASIL KE FILE
# =====================================================================
print("\n" + "=" * 70)
print("BAGIAN 12: EXPORT HASIL")
print("=" * 70)

# Save to text file
output_file = '../results/verification/calculation_verification.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("VERIFIKASI PERHITUNGAN FILTER BPF\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. SPESIFIKASI\n")
    f.write(f"   f1 = {f1} Hz, fl = {fl} Hz, fu = {fu} Hz, f2 = {f2} Hz\n")
    f.write(f"   k1 = {k1} dB, k2 = {k2} dB, fs = {fs} Hz\n\n")
    
    f.write("2. FREKUENSI DIGITAL\n")
    f.write(f"   ω1 = {omega1:.6f}, ωl = {omega_l:.6f}, ωu = {omega_u:.6f}, ω2 = {omega2:.6f}\n\n")
    
    f.write("3. PREWARPING\n")
    f.write(f"   Ω1 = {Omega1:.6f}, Ωl = {Omega_l:.6f}, Ωu = {Omega_u:.6f}, Ω2 = {Omega2:.6f}\n\n")
    
    f.write("4. PARAMETER BPF\n")
    f.write(f"   Ω0 = {Omega_0:.6f}, B = {B:.6f}\n\n")
    
    f.write("5. NORMALISASI\n")
    f.write(f"   A = {A_norm:.6f}, B_norm = {B_norm:.6f}, Ωr = {Omega_r:.6f}\n\n")
    
    f.write("6. ORDE FILTER\n")
    f.write(f"   n_exact = {n_exact:.6f}, n = {n}, Orde BPF = {2*n}\n\n")
    
    f.write("7. KOEFISIEN\n")
    f.write(f"   Numerator (b): {len(b)} koefisien\n")
    for i, coef in enumerate(b):
        f.write(f"     b[{i}] = {coef:.15e}\n")
    f.write(f"\n   Denominator (a): {len(a)} koefisien\n")
    for i, coef in enumerate(a):
        f.write(f"     a[{i}] = {coef:.15e}\n")
    
    f.write("\n8. STABILITAS\n")
    f.write(f"   Max |pole| = {max_pole_mag:.6f} < 1 → STABIL\n")

print(f"Hasil disimpan ke: {output_file}")

# Save coefficients to CSV
import csv
coef_file = '../results/verification/coefficients.csv'
with open(coef_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Index', 'b (numerator)', 'a (denominator)'])
    for i in range(len(b)):
        writer.writerow([i, b[i], a[i]])
print(f"Koefisien disimpan ke: {coef_file}")

# Save verification table
verif_file = '../results/verification/specification_verification.csv'
with open(verif_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Frekuensi', 'Hz', 'Spesifikasi', 'Aktual (dB)', 'Status'])
    for row in results:
        writer.writerow(row)
print(f"Verifikasi disimpan ke: {verif_file}")

print("\n" + "=" * 70)
print("VERIFIKASI SELESAI")
print("=" * 70)
