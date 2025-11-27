# 🔧 Design Steps: Band Pass Filter using Bilinear Transformation

## Design Specifications

Given:
- $f_1 = 770$ Hz (lower stopband edge)
- $f_l = 920$ Hz (lower cutoff frequency)
- $f_u = 1040$ Hz (upper cutoff frequency)
- $f_2 = 1155$ Hz (upper stopband edge)
- $k_1 = 2$ dB (passband ripple)
- $k_2 = 40$ dB (stopband attenuation)
- $f_s = 6000$ Hz (sampling frequency)

---

## Step 1: Convert Analog Frequencies to Digital

**Formula:**
$$\omega = \frac{2\pi f}{f_s}$$

**Calculations:**

$$\omega_1 = \frac{2\pi \times 770}{6000} = 0.806860 \text{ rad/sample}$$

$$\omega_l = \frac{2\pi \times 920}{6000} = 0.963394 \text{ rad/sample}$$

$$\omega_u = \frac{2\pi \times 1040}{6000} = 1.089060 \text{ rad/sample}$$

$$\omega_2 = \frac{2\pi \times 1155}{6000} = 1.209513 \text{ rad/sample}$$

---

## Step 2: Prewarping (Digital to Analog)

**Formula:**
$$\Omega = \frac{2}{T} \tan\left(\frac{\omega}{2}\right)$$

Using $T = 2$ (normalized):

$$\Omega = \tan\left(\frac{\omega}{2}\right)$$

**Calculations:**

$$\Omega_1 = \tan(0.403430) = 0.427025 \text{ rad/s}$$

$$\Omega_l = \tan(0.481697) = 0.523099 \text{ rad/s}$$

$$\Omega_u = \tan(0.544530) = 0.605305 \text{ rad/s}$$

$$\Omega_2 = \tan(0.604756) = 0.696406 \text{ rad/s}$$

---

## Step 3: BPF to Normalized LPF Transformation

### Step 3.1: Calculate BPF Parameters

**Center Frequency:**
$$\Omega_0 = \sqrt{\Omega_l \times \Omega_u} = \sqrt{0.523099 \times 0.605305} = 0.562731 \text{ rad/s}$$

**Bandwidth:**
$$B = \Omega_u - \Omega_l = 0.605305 - 0.523099 = 0.082206 \text{ rad/s}$$

### Step 3.2: Normalize Stopband Frequencies

**Formula:**
$$\Omega_s = \frac{1}{B} \cdot \frac{|\Omega^2 - \Omega_0^2|}{\Omega}$$

**For lower stopband ($\Omega_1$):**
$$\Omega_{s1} = \frac{1}{0.082206} \cdot \frac{|0.427025^2 - 0.562731^2|}{0.427025}$$
$$\Omega_{s1} = 12.168 \cdot \frac{|0.182350 - 0.316666|}{0.427025}$$
$$\Omega_{s1} = 12.168 \times 0.314528 = 3.827$$

**For upper stopband ($\Omega_2$):**
$$\Omega_{s2} = \frac{1}{0.082206} \cdot \frac{|0.696406^2 - 0.562731^2|}{0.696406}$$
$$\Omega_{s2} = 12.168 \cdot \frac{|0.484981 - 0.316666|}{0.696406}$$
$$\Omega_{s2} = 12.168 \times 0.241658 = 2.941$$

**Select more restrictive:**
$$\Omega_s = \min(\Omega_{s1}, \Omega_{s2}) = 2.941$$

---

## Step 4: Calculate Filter Order

**Formula:**
$$n \geq \frac{\log\left(\frac{10^{k_2/10} - 1}{10^{k_1/10} - 1}\right)}{2 \log(\Omega_s)}$$

**Calculations:**

$$\epsilon^2 = 10^{2/10} - 1 = 10^{0.2} - 1 = 1.5849 - 1 = 0.5849$$

$$A^2 = 10^{40/10} - 1 = 10^4 - 1 = 9999$$

$$n \geq \frac{\log(9999/0.5849)}{2 \log(2.941)}$$

$$n \geq \frac{\log(17096.26)}{2 \times 0.4685}$$

$$n \geq \frac{4.233}{0.937} = 4.517$$

**Result:** $n = 5$ (rounded up)

---

## Step 5: Design Normalized Butterworth LPF

For order $n = 5$, the Butterworth poles on the unit circle:

$$s_k = e^{j\pi(2k+n-1)/(2n)} \text{ for } k = 1, 2, ..., n$$

Poles (left half-plane only):

| k | Angle | $s_k$ |
|---|-------|-------|
| 1 | 108° | $-0.309 + j0.951$ |
| 2 | 144° | $-0.809 + j0.588$ |
| 3 | 180° | $-1.000 + j0.000$ |
| 4 | 216° | $-0.809 - j0.588$ |
| 5 | 252° | $-0.309 - j0.951$ |

**Transfer Function:**
$$H_{LPF}(s) = \frac{1}{(s+1)(s^2+0.618s+1)(s^2+1.618s+1)}$$

---

## Step 6: Transform LPF → BPF (Analog)

**Transformation:**
$$s \rightarrow \frac{s^2 + \Omega_0^2}{Bs}$$

This transforms:
- Order $n$ LPF → Order $2n$ BPF
- Single pole → Pair of conjugate poles
- Creates zeros at origin

**Result:** 10th-order analog BPF

---

## Step 7: Bilinear Transformation (Analog → Digital)

**Transformation:**
$$s = \frac{2}{T} \cdot \frac{z-1}{z+1} = \frac{z-1}{z+1} \text{ (for } T=2 \text{)}$$

Apply to $H_a(s)$ to get $H(z)$

**Result:** Digital transfer function $H(z) = \frac{B(z)}{A(z)}$

---

## Step 8: Difference Equation

From $H(z) = \frac{B(z)}{A(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{\sum_{k=0}^{N} a_k z^{-k}}$

The difference equation:

$$y[n] = \sum_{k=0}^{M} b_k \cdot x[n-k] - \sum_{k=1}^{N} a_k \cdot y[n-k]$$

---

## Step 9: Filter Realization

The filter can be realized as:
1. **Direct Form I**: Separate FIR and IIR sections
2. **Direct Form II**: Transposed structure (fewer delays)
3. **Cascade (SOS)**: Second-order sections (most stable)
4. **Parallel**: Sum of first/second-order sections

**Recommended:** Cascade form with second-order sections for numerical stability.

---

## Summary Table

| Step | Description | Result |
|------|-------------|--------|
| 1 | Digital frequencies | $\omega_1 = 0.8069$, $\omega_l = 0.9634$, $\omega_u = 1.0891$, $\omega_2 = 1.2095$ |
| 2 | Prewarped frequencies | $\Omega_1 = 0.4270$, $\Omega_l = 0.5231$, $\Omega_u = 0.6053$, $\Omega_2 = 0.6964$ |
| 3 | BPF parameters | $\Omega_0 = 0.5627$, $B = 0.0822$ |
| 4 | Normalized stopband | $\Omega_s = 2.941$ |
| 5 | Filter order | $n = 5$ (BPF order = 10) |
| 6 | Design complete | Transfer function $H(z)$ |

---

## Verification Checklist

- [ ] Magnitude at $f_1$ (770 Hz) ≤ -40 dB
- [ ] Magnitude at $f_l$ (920 Hz) ≥ -2 dB
- [ ] Magnitude at $f_u$ (1040 Hz) ≥ -2 dB
- [ ] Magnitude at $f_2$ (1155 Hz) ≤ -40 dB
- [ ] All poles inside unit circle (stable)
- [ ] Passband ripple ≤ 2 dB
