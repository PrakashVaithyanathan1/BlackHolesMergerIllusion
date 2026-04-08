import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from scipy.stats import pearsonr

# Configuration for GW150914 (The First Discovery)
t0 = event_gps('GW150914')
stages = [
    (t0-110, t0-10, "Pre-Merger Baseline", 'gray'),
    (t0-1,   t0+1,  "During Merger", 'blue'),
    (t0+1,   t0+11, "10s Post-Merger", 'green'),
    (t0+90,  t0+140,"50s Post-Merger", 'red')
]

print(f"--- ANALYZING GW150914: METRIC PERSISTENCE VALIDATION ---")

# 1. EMPIRICAL BACKGROUND ESTIMATION (TIME-SLIDES)
print("Performing 1,000 Time-Slides for Background Estimation...")
h1_bg_full = TimeSeries.fetch_open_data('H1', t0-110, t0-10, sample_rate=16384).highpass(1000)
l1_bg_full = TimeSeries.fetch_open_data('L1', t0-110, t0-10, sample_rate=16384).highpass(1000)

num_slides = 1000
slide_rs = []
for i in range(num_slides):
    shift = int((0.1 + i * 0.05) * 16384) # Incremental shifts > 10ms
    h1_shifted = np.roll(h1_bg_full.value[:32768], shift)
    r_bg, _ = pearsonr(h1_shifted, l1_bg_full.value[:32768])
    slide_rs.append(r_bg)

bg_mu, bg_std = np.mean(slide_rs), np.std(slide_rs)

# 2. MAIN ANALYSIS & PLOTTING
fig, ax = plt.subplots(figsize=(12, 7))

for start, end, label, color in stages:
    print(f"Processing {label}...")
    h1 = TimeSeries.fetch_open_data('H1', start, end, sample_rate=16384).highpass(1000)
    l1 = TimeSeries.fetch_open_data('L1', start, end, sample_rate=16384).highpass(1000)

    r_obs, _ = pearsonr(h1.value, l1.value)
    sigma_emp = (r_obs - bg_mu) / bg_std # Empirical significance

    psd = h1.psd(fftlength=1, method='median')
    ax.loglog(psd, label=f"{label} (r={r_obs:.4f}, $\sigma_{{emp}}$={sigma_emp:.1f})", color=color, alpha=0.6)

# ABDM Stability Floor (Universal Constant α=1.024)
f = np.logspace(3, 3.2, 100)
ax.loglog(f, 1e-45 * (f**-1.024), 'k--', label='ABDM Floor (α=1.024)', linewidth=2)

ax.set_title("GW150914: High-Frequency Metric & Achromatic Shield Analysis", fontsize=14)
ax.set_xlabel("Frequency [Hz]", fontsize=12)
ax.set_ylabel("Strain", fontsize=12)
ax.set_xlim(1000, 1600)
ax.set_ylim(1e-50, 1e-40)

# KEEPS DATA BOX AT BOTTOM LEFT
ax.legend(loc='lower left', frameon=True, shadow=True)

plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

# SAVES UNIQUE OUTPUT FOR SECOND EVENT
output_filename = 'GW150914_metric_validation.pdf'
plt.savefig(output_filename)
print(f"--- SUCCESS: Figure saved as {output_filename} ---")
plt.show()
