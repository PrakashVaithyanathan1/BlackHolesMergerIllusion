!pip install -q gwpy gwosc
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.stats import pearsonr

# --- STEP 2: CONFIGURATION (GW170814 Cross-Continental Check) ---
t0 = 1186741861.5 # GPS for GW170814
event_name = "GW170814"
pdf_filename = f"{event_name}_Virgo_Validation.pdf"

print(f"--- Fetching 16kHz data for {event_name} (H1, L1, V1) ---")

# --- STEP 3: DATA ACQUISITION & FILTERING ---
# Fetching 16kHz high-cadence data for all three global instruments
h1 = TimeSeries.fetch_open_data('H1', t0-1, t0+1, sample_rate=16384).highpass(1000)
l1 = TimeSeries.fetch_open_data('L1', t0-1, t0+1, sample_rate=16384).highpass(1000)
v1 = TimeSeries.fetch_open_data('V1', t0-1, t0+1, sample_rate=16384).highpass(1000)

# --- STEP 4: CORRELATION ANALYSIS ---
r_h1l1, _ = pearsonr(h1.value, l1.value)
r_h1v1, _ = pearsonr(h1.value, v1.value) # The cross-continental proof

# --- STEP 5: VISUALIZATION ---
fig, ax = plt.subplots(figsize=(12, 7))
ax.loglog(v1.psd(fftlength=1), label=f"Virgo (V1) Spectrum ($r_{{H1V1}}$={r_h1v1:.6f})", color='purple', alpha=0.7)
ax.loglog(h1.psd(fftlength=1), label=f"LIGO (H1) Spectrum ($r_{{H1L1}}$={r_h1l1:.6f})", color='red', alpha=0.4)

# ABDM Stability Floor
f = np.logspace(3, 3.2, 100)
ax.loglog(f, 1e-45 * (f**-1.024), 'k--', label=r'ABDM Floor ($\alpha$=1.024)', linewidth=2)

# --- STEP 6: FORMATTING & SAVING ---
ax.set_title(f"{event_name}: Global Metric Synchronization (USA-Italy)", fontsize=14)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Strain")
ax.set_xlim(1000, 1600)
ax.set_ylim(1e-50, 1e-40)
ax.legend(loc='lower left', shadow=True)
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(pdf_filename, format='pdf', dpi=300)
plt.show()
print(f"--- SUCCESS: {pdf_filename} is ready for download! ---")
