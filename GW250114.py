!pip install -q gwpy gwosc

import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.stats import pearsonr

# Configuration for GW250114
t0 = 1420878141.2
event_name = "GW250114"

stages = [
    (t0-110, t0-10, "Pre-Merger Baseline", 'gray'),
    (t0-1,   t0+1,  "During Merger", 'blue'),
    (t0+1,   t0+11, "10s Post-Merger", 'green'),
    (t0+90,  t0+140, "50s Post-Merger", 'red')
]

print(f"--- STARTING VALIDATED ANALYSIS FOR {event_name} ---")

# Fetching high-res 16kHz data
h1_bg_full = TimeSeries.fetch_open_data('H1', t0-110, t0-10, sample_rate=16384).highpass(1000)
l1_bg_full = TimeSeries.fetch_open_data('L1', t0-110, t0-10, sample_rate=16384).highpass(1000)

num_slides = 1000
slide_rs = []
for i in range(num_slides):
    shift = int((0.1 + i * 0.05) * 16384)
    h1_shifted = np.roll(h1_bg_full.value[:32768], shift)
    r_bg, _ = pearsonr(h1_shifted, l1_bg_full.value[:32768])
    slide_rs.append(r_bg)

background_mu = np.mean(slide_rs)
background_std = np.std(slide_rs)

fig, ax = plt.subplots(figsize=(14, 7))

for start, end, label, color in stages:
    h1 = TimeSeries.fetch_open_data('H1', start, end, sample_rate=16384).highpass(1000)
    l1 = TimeSeries.fetch_open_data('L1', start, end, sample_rate=16384).highpass(1000)
    r_obs, _ = pearsonr(h1.value, l1.value)
    sigma_emp = (r_obs - background_mu) / background_std
    psd = h1.psd(fftlength=1, method='median')
    # Using 'r' for raw string to handle LaTeX sigma correctly
    ax.loglog(psd, label=fr"{label} (r={r_obs:.4f}, $\sigma$={sigma_emp:.1f})", color=color, alpha=0.6)

# ABDM Stability Floor
f = np.logspace(3, 3.204, 100)
ax.loglog(f, 1e-45 * (f**-1.024), 'k--', label=r'ABDM Floor ($\alpha$=1.024)', linewidth=2)

ax.set_title(f"{event_name}: Metric Lifecycle & Achromatic Shield", fontsize=14)
ax.set_xlabel("Frequency [Hz]", fontsize=12)
ax.set_ylabel("Strain", fontsize=12)
ax.set_xlim(1000, 1600)
ax.set_ylim(1e-50, 1e-38)

# Legend placed outside for clarity
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, shadow=True)

plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

# Save output to PDF
output_file = f"{event_name}_Metric_Validation.pdf"
plt.savefig(output_file, bbox_inches='tight')
plt.show()

print(f"--- SUCCESS: PDF saved as {output_file} ---")
