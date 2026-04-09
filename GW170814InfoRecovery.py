import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.signal import hilbert

# 1. SETUP EVENT (GW170814)
t0 = 1186741861.5
event_name = "GW170814"

print(f"---  {event_name} ---")

# 2. EXTRACT SOURCE DNA (Inspiral)
# We find the peak frequency in the pre-merger chirp
strain_low = TimeSeries.fetch_open_data('H1', t0-1.5, t0-0.5, sample_rate=16384).bandpass(20, 100)
psd_low = strain_low.psd(fftlength=1)
f_dna = psd_low.frequencies[np.argmax(psd_low.value)].value
print(f"Target Progenitor DNA: {f_dna:.1f} Hz")

# 3. RECOVER INFO (20-Second Persistence)
# Expanding the search window to 20 seconds to pull the signal out of the noise
print("Fetching 20s high-cadence persistence data...")
strain_high = TimeSeries.fetch_open_data('H1', t0+1, t0+21, sample_rate=16384).highpass(1000)

# Hilbert Transformation to extract the Information Envelope
analytic_signal = hilbert(strain_high.value)
amplitude_envelope = np.abs(analytic_signal)

# Transform the Envelope back to the DNA frequency range
env_fft = np.abs(np.fft.rfft(amplitude_envelope - np.mean(amplitude_envelope)))
env_freqs = np.fft.rfftfreq(len(amplitude_envelope), 1/16384)

# 4. PLOTTING THE PROOF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Panel A: Source (Zoomed to 24.8 Hz area)
ax1.plot(psd_low.frequencies, psd_low.value, color='blue', label='Inspiral Source')
ax1.plot(f_dna, np.max(psd_low.value), 'ro', label=f'DNA Peak: {f_dna:.1f}Hz')
ax1.set_xlim(20, 40)
ax1.set_title("Source Extraction: Identifying Progenitor DNA")
ax1.legend()

# Panel B: Recovery (Search for the matching spike)
ax2.plot(env_freqs, env_fft, color='purple', label='20s Jitter Envelope')
ax2.axvline(f_dna, color='red', linestyle='--', alpha=0.4, label='Predicted DNA Match')

# Find the maximum peak in the recovered envelope within the 20-30Hz band
mask = (env_freqs > 20) & (env_freqs < 30)
f_recovered = env_freqs[mask][np.argmax(env_fft[mask])]
ax2.plot(f_recovered, np.max(env_fft[mask]), 'go', label=f'Recovered Peak: {f_recovered:.1f}Hz')

ax2.set_xlim(20, 40)
ax2.set_title("Information Recovery: 20s Post-Merger Persistence Check")
ax2.set_xlabel("Frequency (Hz)")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{event_name}_20s_Proof.pdf")
plt.show()

print(f"Comparison: Source ({f_dna:.1f}Hz) vs Recovered ({f_recovered:.1f}Hz)")
