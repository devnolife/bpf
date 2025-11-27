# 📚 Theory: Band Pass Filter Design

## 1. Signal Processing Basics

### 1.1 Frequency Domain Representation

Signals can be represented in two domains:
- **Time Domain**: Signal amplitude vs time, $x(t)$
- **Frequency Domain**: Signal amplitude vs frequency, $X(f)$ or $X(\omega)$

The relationship between these domains is given by the **Fourier Transform**:

$$X(f) = \int_{-\infty}^{\infty} x(t) e^{-j2\pi ft} dt$$

And the inverse:

$$x(t) = \int_{-\infty}^{\infty} X(f) e^{j2\pi ft} df$$

### 1.2 Sampling Theorem (Nyquist Criteria)

For a continuous signal to be accurately represented in digital form:

$$f_s > 2 \cdot f_{max}$$

Where:
- $f_s$ = Sampling frequency
- $f_{max}$ = Maximum frequency in the signal

The **Nyquist frequency** is $f_N = f_s / 2$

### 1.3 Analog vs Digital Signals

| Property | Analog | Digital |
|----------|--------|---------|
| Time | Continuous | Discrete (samples) |
| Amplitude | Continuous | Quantized |
| Representation | $x(t)$ | $x[n]$ |
| Frequency | $f$ (Hz) or $\Omega$ (rad/s) | $\omega$ (rad/sample) |

---

## 2. Filter Theory

### 2.1 Filter Types

| Type | Function | Passes | Blocks |
|------|----------|--------|--------|
| **LPF** (Low Pass) | Removes high frequencies | $f < f_c$ | $f > f_c$ |
| **HPF** (High Pass) | Removes low frequencies | $f > f_c$ | $f < f_c$ |
| **BPF** (Band Pass) | Passes frequency band | $f_l < f < f_u$ | $f < f_l$ and $f > f_u$ |
| **BSF** (Band Stop) | Removes frequency band | $f < f_l$ and $f > f_u$ | $f_l < f < f_u$ |

### 2.2 Filter Specifications

For a Band Pass Filter:

```
          |
      0dB |----+     +----+     +----
          |    |     |    |     |
    -k1dB |    +-----+    +-----+    Passband
          |    |     |    |     |
          |    |     |    |     |
    -k2dB +----+     |    |     +---- Stopband
          |         |    |          
          +---+-----+----+-----+----> Frequency
              f1   fl   fu   f2
```

Where:
- $f_1$ = Lower stopband edge
- $f_l$ = Lower cutoff (passband edge)
- $f_u$ = Upper cutoff (passband edge)
- $f_2$ = Upper stopband edge
- $k_1$ = Passband ripple (dB)
- $k_2$ = Stopband attenuation (dB)

### 2.3 Magnitude and Phase Response

The frequency response of a filter:

$$H(j\omega) = |H(j\omega)| \cdot e^{j\phi(\omega)}$$

- **Magnitude Response**: $|H(j\omega)|$ - How much each frequency is amplified/attenuated
- **Phase Response**: $\phi(\omega)$ - Phase shift at each frequency
- **Group Delay**: $\tau_g(\omega) = -\frac{d\phi(\omega)}{d\omega}$

### 2.4 IIR vs FIR Filters

| Property | IIR | FIR |
|----------|-----|-----|
| Full Name | Infinite Impulse Response | Finite Impulse Response |
| Recursion | Yes (feedback) | No |
| Poles | Has poles (inside unit circle) | Only zeros |
| Stability | Must be checked | Always stable |
| Phase | Non-linear | Can be linear |
| Order | Lower for same specs | Higher |
| Implementation | Difference equation | Convolution |

---

## 3. Butterworth Filter Design

### 3.1 Butterworth Characteristics

The Butterworth filter is characterized by a **maximally flat magnitude response** in the passband.

Magnitude squared response:

$$|H(j\Omega)|^2 = \frac{1}{1 + (\Omega/\Omega_c)^{2n}}$$

Where:
- $n$ = Filter order
- $\Omega_c$ = Cutoff frequency

Properties:
- Monotonic response (no ripples)
- -3dB at cutoff frequency
- Roll-off rate: -20n dB/decade

### 3.2 Butterworth Poles

The poles of a Butterworth LPF lie on a circle in the s-plane:

$$s_k = \Omega_c \cdot e^{j\pi(2k+n-1)/(2n)}$$

for $k = 1, 2, ..., n$

Only left-half plane poles are used for stability.

### 3.3 Order Calculation

Given passband ripple ($k_1$) and stopband attenuation ($k_2$):

$$n \geq \frac{\log\left(\frac{10^{k_2/10} - 1}{10^{k_1/10} - 1}\right)}{2 \log(\Omega_s)}$$

Where $\Omega_s$ is the normalized stopband frequency.

### 3.4 Frequency Transformations

To convert a normalized LPF to other filter types:

| Transform | Substitution |
|-----------|--------------|
| LPF → HPF | $s \rightarrow \Omega_c/s$ |
| LPF → BPF | $s \rightarrow (s^2 + \Omega_0^2)/(Bs)$ |
| LPF → BSF | $s \rightarrow Bs/(s^2 + \Omega_0^2)$ |

For BPF:
- $\Omega_0 = \sqrt{\Omega_l \cdot \Omega_u}$ (center frequency)
- $B = \Omega_u - \Omega_l$ (bandwidth)

---

## 4. Bilinear Transformation

### 4.1 Overview

The bilinear transformation converts an analog filter $H_a(s)$ to a digital filter $H(z)$:

$$s = \frac{2}{T} \cdot \frac{z-1}{z+1}$$

Or equivalently:

$$z = \frac{1 + sT/2}{1 - sT/2}$$

### 4.2 Frequency Warping

The bilinear transformation introduces frequency warping:

$$\Omega = \frac{2}{T} \tan\left(\frac{\omega}{2}\right)$$

Where:
- $\Omega$ = Analog frequency (rad/s)
- $\omega$ = Digital frequency (rad/sample)
- $T$ = Sampling period

### 4.3 Prewarping

To compensate for frequency warping, we prewarp the specifications:

$$\Omega' = \frac{2}{T} \tan\left(\frac{\omega}{2}\right)$$

This ensures critical frequencies match after transformation.

---

## 5. Design Steps Summary

1. **Specify** digital filter requirements ($f_1, f_l, f_u, f_2, k_1, k_2, f_s$)
2. **Convert** to digital frequencies: $\omega = 2\pi f/f_s$
3. **Prewarp** to analog: $\Omega = (2/T)\tan(\omega/2)$
4. **Calculate** BPF parameters: $\Omega_0$, $B$
5. **Normalize** to LPF: $\Omega_s$
6. **Determine** filter order $n$
7. **Design** normalized Butterworth LPF
8. **Transform** LPF → BPF (analog)
9. **Apply** bilinear transformation → Digital BPF
10. **Verify** specifications

---

## References

1. Oppenheim, A.V., Schafer, R.W., "Discrete-Time Signal Processing"
2. Proakis, J.G., Manolakis, D.G., "Digital Signal Processing"
3. Parks, T.W., Burrus, C.S., "Digital Filter Design"
