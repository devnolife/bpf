# Band Pass Filter Design Project

## 📋 Overview

This project implements a digital **Band Pass Filter (BPF)** using **Butterworth approximation** and **bilinear transformation**. It includes complete theory, Python implementation, visualizations, and analysis.

## 🎯 Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| f₁ | 770 Hz | Lower stopband edge |
| fₗ | 920 Hz | Lower cutoff frequency |
| fᵤ | 1040 Hz | Upper cutoff frequency |
| f₂ | 1155 Hz | Upper stopband edge |
| k₁ | 2 dB | Passband ripple |
| k₂ | 40 dB | Stopband attenuation |
| fₛ | 6000 Hz | Sampling frequency |

## 📁 Project Structure

```
BPF_Filter_Design/
│
├── docs/                     # Documentation
│   ├── theory.md            # Signal processing theory
│   └── design_steps.md      # Step-by-step design process
│
├── src/                      # Source code
│   ├── __init__.py          # Package initialization
│   ├── filter_helpers.py    # Helper functions
│   ├── filter_design.py     # Filter design functions
│   ├── filter_analysis.py   # Visualization tools
│   ├── signal_processing.py # Signal generation & filtering
│   └── main_bpf_design.py   # Main workflow script
│
├── notebooks/                # Jupyter notebooks
│   └── BPF_Design_Complete.ipynb
│
├── results/                  # Output files
│   ├── plots/               # Generated plots
│   ├── data/                # Filter coefficients
│   └── screenshots/         # Screenshots
│
├── tests/                    # Unit tests
│   └── test_filter.py
│
├── requirements.txt          # Python dependencies
├── plan.md                   # Project plan
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Main Script

```bash
cd src
python main_bpf_design.py
```

### 3. Run Jupyter Notebook

```bash
jupyter notebook notebooks/BPF_Design_Complete.ipynb
```

### 4. Run Tests

```bash
cd tests
python -m pytest test_filter.py -v
```

## 📊 Generated Outputs

The main script generates:

1. **Filter Coefficients** (`results/data/filter_coefficients.txt`)
2. **Plots** (`results/plots/`):
   - `specifications.png` - Filter template
   - `magnitude_response.png` - Magnitude response
   - `phase_response.png` - Phase response
   - `pole_zero.png` - Pole-zero diagram
   - `impulse_response.png` - h[n]
   - `step_response.png` - Step response
   - `group_delay.png` - Group delay
   - `input_output_comparison.png` - Signal filtering demo
   - `frequency_comparison.png` - Spectrum comparison

## 📐 Design Method

1. **Convert** analog frequencies to digital (ω = 2πf/fs)
2. **Prewarp** to analog domain (Ω = tan(ω/2))
3. **Calculate** BPF parameters (Ω₀, B)
4. **Normalize** to LPF (Ωₛ)
5. **Determine** filter order (n)
6. **Design** Butterworth prototype
7. **Transform** LPF → BPF
8. **Apply** bilinear transformation

## 🔧 Key Functions

### filter_helpers.py
- `analog_to_digital()` - Hz to rad/sample
- `prewarping()` - Digital to analog
- `calculate_order()` - Butterworth order

### filter_design.py
- `design_butterworth_bpf()` - Main design function
- `get_difference_equation()` - Generate difference equation

### filter_analysis.py
- `plot_magnitude_response()` - Magnitude plot
- `plot_pole_zero()` - Pole-zero diagram
- `verify_specifications()` - Check specs

### signal_processing.py
- `generate_test_signal()` - Multi-frequency signal
- `apply_filter()` - Filter application
- `fft_analysis()` - Frequency analysis

## 📚 Theory

See `docs/theory.md` for:
- Signal processing basics
- Filter theory (IIR vs FIR)
- Butterworth characteristics
- Bilinear transformation

## ✅ Verification

The filter is verified to meet all specifications:
- ✓ Stopband attenuation ≥ 40 dB at f₁ and f₂
- ✓ Passband ripple ≤ 2 dB between fₗ and fᵤ
- ✓ All poles inside unit circle (stable)

## 📝 License

This project is for educational purposes.

## 👤 Author

BPF Design Project - 2024
