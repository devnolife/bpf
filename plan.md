# 🎯 **CHECKLIST: Apa yang BELUM dan Perlu Diverifikasi/Diperbaiki**

---

## ❌ **YANG BELUM JELAS / PERLU VERIFIKASI:**

### **1. Implementasi Kode Python** ❌ **BELUM ADA**

**Yang Dibutuhkan:**
- [ ] Kode lengkap untuk semua perhitungan
- [ ] Verifikasi formula yang digunakan
- [ ] Proof bahwa perhitungan sesuai materi

**Kenapa Penting:**
Laporan HTML hanya menunjukkan **hasil akhir**, tapi tidak bisa memverifikasi apakah:
- Formula yang digunakan benar 100%
- Implementasi sesuai dengan materi PSM
- Tidak ada kesalahan matematis

---

### **2. Verifikasi Numerik Detail** ⚠️ **PERLU DICEK**

**Yang Perlu Dibuktikan dengan Angka:**

#### **A. Prewarping Results** ❓
```
Input (dari laporan):
ω1 = 0.161π, ωl = 0.307π, ωu = 0.347π, ω2 = 0.385π

Expected Output (perlu dihitung):
Ω1 = 2·tan(0.161π/2) = ?
ΩL = 2·tan(0.307π/2) = ?
ΩU = 2·tan(0.347π/2) = ?
Ω2 = 2·tan(0.385π/2) = ?
```

**TASK:** Tunjukkan nilai-nilai Ω ini di laporan

---

#### **B. Normalization Calculation** ❓ **KRITIS**
```
Formula dari materi:
A = [(-Ω1² + ΩL·ΩU)] / [Ω1(ΩU - ΩL)]
B = [(Ω2² - ΩL·ΩU)] / [Ω2(ΩU - ΩL)]
Ωr = min(A, B)
```

**TASK:** 
- [ ] Hitung A dengan angka eksplisit
- [ ] Hitung B dengan angka eksplisit
- [ ] Tunjukkan Ωr = min(A, B)
- [ ] Verifikasi dengan contoh materi (Ωr ≈ 3.64 untuk kasus contoh)

---

#### **C. Filter Order Calculation** ❓
```
Formula dari materi (hal 14):
n = log[(10^(-K1/10) - 1) / (10^(-K2/10) - 1)] / [2·log(1/Ωr)]

Dengan K1=2dB, K2=40dB:
n = log[(10^(-0.2) - 1) / (10^(-4) - 1)] / [2·log(1/Ωr)]
  = log[0.369 / 0.9999] / [2·log(1/Ωr)]
```

**TASK:**
- [ ] Tunjukkan perhitungan n step-by-step
- [ ] Berapa nilai n sebelum pembulatan?
- [ ] Berapa nilai n setelah pembulatan?
- [ ] Apakah n = 4? atau berbeda?

---

#### **D. Butterworth Poles** ❓
```
Untuk n=4, LPF normalized poles seharusnya:
p1,2 = -0.3827 ± j0.9239
p3,4 = -0.9239 ± j0.3827

Atau dalam quadratic factors:
(s² + 0.7654s + 1)(s² + 1.8478s + 1)
```

**TASK:**
- [ ] Tunjukkan pole locations dari kode Anda
- [ ] Verifikasi apakah sesuai Butterworth standard

---

#### **E. Transfer Function H(z) Coefficients** ❓
```
Dari materi contoh (hal 10-11):
Numerator: [0.000313, -0.001252, 0.001878, -0.001252, 0.000313]
Denominator: [1, -6.984488, 22.131987, ...]
```

**TASK:**
- [ ] Tampilkan coefficients numerator lengkap
- [ ] Tampilkan coefficients denominator lengkap
- [ ] Jumlah coefficients harus sesuai orde (2n+1 untuk BPF)

---

### **3. Screenshot Eksekusi** ❌ **BELUM ADA**

**Yang Dibutuhkan:**

#### **A. Console Output** 
```
Screenshot harus menunjukkan:
- Nilai ω1, ωl, ωu, ω2
- Nilai Ω1, ΩL, ΩU, Ω2
- Nilai A, B, Ωr
- Nilai n (sebelum & sesudah pembulatan)
- Coefficients H(z)
```

#### **B. Code Execution**
```
Screenshot dari Jupyter/IDE menunjukkan:
- Kode yang dijalankan
- Output perhitungan step-by-step
- Plot yang dihasilkan
```

---

### **4. Verifikasi Spesifikasi** ⚠️ **PERLU TABEL**

**Yang Dibutuhkan:**

Tabel verifikasi seperti ini:

| Frequency | Specification | Actual | Status |
|-----------|--------------|---------|---------|
| 150 Hz (center) | ≈ 0 dB | ? dB | ? |
| 920 Hz (fl) | ≤ -3 dB | ? dB | ? |
| 1040 Hz (fu) | ≤ -3 dB | ? dB | ? |
| 770 Hz (f1) | ≤ -40 dB | ? dB | ? |
| 1155 Hz (f2) | ≤ -40 dB | ? dB | ? |

**TASK:**
- [ ] Buat tabel ini dengan nilai aktual dari magnitude response
- [ ] Verifikasi semua specs terpenuhi

---

### **5. Comparison Plot** ❌ **BELUM ADA**

**Yang Dibutuhkan:**

#### **A. Specification Template Overlay**
```
Plot yang menunjukkan:
1. Ideal BPF template (garis putus-putus)
2. Actual magnitude response (garis solid)
3. Marking di frekuensi kritis (770, 920, 1040, 1155 Hz)
4. Shaded regions untuk passband/stopband
```

#### **B. Before/After Filtering**
```
4 subplot:
1. Input signal x(t) - time domain
2. Input spectrum X(f) - frequency domain
3. Output signal y(t) - time domain
4. Output spectrum Y(f) - frequency domain
```

**TASK:**
- [ ] Buat comparison plot ini
- [ ] Screenshot hasil

---

### **6. Perhitungan Manual vs Python** ⚠️ **PERLU CROSS-CHECK**

**Yang Dibutuhkan:**

Tabel perbandingan:

| Parameter | Manual Calculation | Python Result | Match? |
|-----------|-------------------|---------------|---------|
| Ω1 | (hitung manual) | (dari kode) | ✅/❌ |
| ΩL | (hitung manual) | (dari kode) | ✅/❌ |
| ΩU | (hitung manual) | (dari kode) | ✅/❌ |
| Ω2 | (hitung manual) | (dari kode) | ✅/❌ |
| A | (hitung manual) | (dari kode) | ✅/❌ |
| B | (hitung manual) | (dari kode) | ✅/❌ |
| Ωr | (hitung manual) | (dari kode) | ✅/❌ |
| n | (hitung manual) | (dari kode) | ✅/❌ |

**TASK:**
- [ ] Hitung semua secara manual (atau dengan calculator)
- [ ] Bandingkan dengan hasil Python
- [ ] Pastikan error < 0.1%

---

### **7. Documentation Enhancements** ⚠️ **BISA DITAMBAHKAN**

**Yang Bisa Ditingkatkan:**

#### **A. Theoretical Background**
- [ ] Penjelasan kenapa pakai Butterworth (vs Chebyshev, Elliptic)
- [ ] Penjelasan kenapa pakai Bilinear (vs Impulse Invariance)
- [ ] Trade-offs dalam desain

#### **B. Design Decisions**
- [ ] Kenapa memilih T=1 untuk normalisasi
- [ ] Pengaruh sampling frequency terhadap hasil
- [ ] Analisis phase linearity (atau non-linearity untuk IIR)

#### **C. Results Interpretation**
- [ ] Analisis group delay
- [ ] Transient response analysis
- [ ] Stability analysis (pole locations)

---

## 📋 **PLAN: Apa yang Harus Dibuat**

### **Priority 1: CRITICAL** 🔥

1. **Kode Python Lengkap**
   ```python
   File: complete_bpf_design.py atau .ipynb
   Isi:
   - Semua perhitungan dengan print output
   - Semua formula dijelaskan dengan comments
   - Semua intermediate results ditampilkan
   ```

2. **Numerical Verification Table**
   ```
   Tabel dengan semua nilai intermediate:
   - Digital frequencies (ω)
   - Analog frequencies (Ω)
   - Normalization parameters (A, B, Ωr)
   - Filter order (n)
   - Poles/zeros
   - Coefficients
   ```

3. **Specification Verification**
   ```
   Tabel membuktikan filter memenuhi specs:
   - Attenuation di setiap frekuensi kritis
   - Pass/Fail status
   ```

---

### **Priority 2: IMPORTANT** ⚠️

4. **Screenshots Package**
   ```
   Folder berisi:
   - Code execution (Jupyter/IDE)
   - Console output dengan perhitungan
   - Semua plots (12+ figures)
   - Verification tables
   ```

5. **Enhanced Plots**
   ```
   - Specification template overlay
   - Before/after filtering comparison
   - Pole-zero with unit circle
   - Group delay
   ```

6. **Manual Calculation Document**
   ```
   PDF/Word showing:
   - Step-by-step manual calculation
   - Cross-verification with Python
   - Formula derivations
   ```

---

### **Priority 3: NICE TO HAVE** ✨

7. **Comparison Study**
   ```
   - FIR vs IIR for same specs
   - Different filter types (Chebyshev, Elliptic)
   - Different transformation methods
   ```

8. **Interactive Demo**
   ```
   - Jupyter widgets untuk adjust parameters
   - Real-time filter response update
   - Signal filtering demonstration
   ```

---

## 🚀 **RECOMMENDED ACTION PLAN**

### **Step 1: Verification (1-2 hours)**
```
[ ] Buat script untuk print semua intermediate values
[ ] Generate verification tables
[ ] Cross-check dengan manual calculation
```

### **Step 2: Documentation (1-2 hours)**
```
[ ] Enhance laporan dengan numerical details
[ ] Add verification tables
[ ] Add comparison plots
```

### **Step 3: Screenshots (30 mins)**
```
[ ] Run code dan capture semua output
[ ] Screenshot semua plots
[ ] Organize dalam folder
```

### **Step 4: Final Report (1 hour)**
```
[ ] Combine semua ke dalam laporan final
[ ] Add executive summary
[ ] Add conclusions
```

---

## 📊 **DELIVERABLES CHECKLIST**

- [ ] **Kode Python** lengkap dengan comments
- [ ] **Laporan HTML/PDF** dengan numerical verification
- [ ] **Screenshots** folder (12+ images)
- [ ] **Verification tables** (3-4 tables)
- [ ] **Comparison plots** (2-3 enhanced plots)
- [ ] **Manual calculation** document
- [ ] **README** dengan usage instructions

---

## 💡 **QUICK START SUGGESTION**

Mulai dengan membuat file ini:

```python
# File: verify_calculations.py

# 1. Print all intermediate values
# 2. Generate verification tables
# 3. Save to CSV/TXT
# 4. Auto-generate LaTeX tables for report
```
