f1 = 500 Hz    # Lower stopband edge
fl = 900 Hz    # Lower cutoff frequency  
fu = 1180 Hz   # Upper cutoff frequency
f2 = 1380 Hz   # Upper stopband edge
k1 = 1 dB      # Passband ripple
k2 = 30 dB     # Stopband attenuation
fs = 9000 Hz   # Sampling frequency
```

---

## 📂 **PROJECT STRUCTURE**
```
BPF_Filter_Design/
│
├── src/
│   ├── 01_helper_functions.py
│   ├── 02_filter_design.py
│   ├── 03_visualization.py
│   ├── 04_signal_processing.py
│   └── 05_main_execution.py
│
├── results/
│   ├── plots/
│   ├── data/
│   └── screenshots/
│
├── docs/
│   └── report.html
│
└── requirements.txt
