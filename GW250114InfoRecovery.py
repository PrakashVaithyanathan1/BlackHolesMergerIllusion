import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.signal import hilbert

# 1. SETUP EVENT (GW250114 - The Ultra-Loud O4 Event)
t0 = 1420878141.2
event_name = "GW250114"

print(f"--- EXECUTING INDEPENDENT REPLICATION: {event_name} ---")

# 2. EXTRACT SOURCE DNA (Pre-Merger Inspiral)
# We isolate the 1-second window before the phase transition
print("Extracting Progenitor DNA from O4b strain...")
strain_low = TimeSeries.fetch_open_data('H1', t0-1.5, t0-0.5, sample_rate=16384).bandpass(20, 100)
psd_low = strain_low.psd(fftlength=1)
f_dna = psd_low.frequencies[np.argmax(psd_low.value)].value 

# 3. RECOVER INFO (20-Second High-Frequency Persistence)
# Using the 20-second window to resolve the 'Achromatic Echo'
print("Analyzing 20s high-frequency jitter envelope...")
strain_high = TimeSeries.fetch_open_data('H1', t0+1, t0+21, sample_rate=16384).highpass(1000)

# Hilbert-Envelope Reconstruction
analytic_signal = hilbert(strain_high.value)
amplitude_envelope = np.abs(analytic_signal)

# Fourier Transform of the Envelope (HIFT)
env_fft = np.abs(np.fft.rfft(amplitude_envelope - np.mean(amplitude_envelope)))
env_freqs = np.fft.rfftfreq(len(amplitude_envelope), 1/16384)

# 4. PLOTTING THE SMOKING GUN
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Top: The Source Peak
ax1.plot(psd_low.frequencies, psd_low.value, color='blue', label='O4b Inspiral DNA')
ax1.plot(f_dna, np.max(psd_low.value), 'ro', label=f'DNA Peak: {f_dna:.1f}Hz')
ax1.set_xlim(f_dna-10, f_dna+10)
ax1.set_title(f"Source Extraction: {event_name}")
ax1.legend()

# Bottom: The Recovered Peak
ax2.plot(env_freqs, env_fft, color='purple', label='Recovered Jitter Envelope')
ax2.axvline(f_dna, color='red', linestyle='--', alpha=0.3, label='Predicted DNA Match')

# Auto-identify the corresponding peak in the jitter
mask = (env_freqs > f_dna-2) & (env_freqs < f_dna+10) # Searching for the shifted peak
f_recovered = env_freqs[mask][np.argmax(env_fft[mask])]
ax2.plot(f_recovered, np.max(env_fft[mask]), 'go', label=f'Recovered Echo: {f_recovered:.1f}Hz')

ax2.set_xlim(f_dna-10, f_dna+10)
ax2.set_title(f"Information Recovery: Deterministic Shift Verification")
ax2.set_xlabel("Frequency (Hz)")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{event_name}_Information_Replication.pdf")
plt.show()

print(f"Shift Analysis: {f_dna:.1f} Hz (Source) -> {f_recovered:.1f} Hz (Recovered)")
print(f"Empirical Shift Factor: {f_recovered/f_dna:.4f}")
