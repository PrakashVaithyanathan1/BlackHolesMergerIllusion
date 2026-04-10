import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from scipy.stats import pearsonr

# CONFIGURATION - Updated with the full name for the last event
events = [
    ('GW150914', 'Discovery'),
    ('GW170814', 'Standard'),
    ('GW190521', 'Massive'),
    ('GW190412', 'Asymmetric'),
    ('GW250114_082203', 'Ultra-Loud') # FIXED NAME
]
fs = 16384
master_data = []

for name, e_type in events:
    print(f"\n--- DEEP SCAN: {name} ({e_type}) ---")
    try:
        t0 = event_gps(name)
        
        # 1. FETCH FULL BLOCKS
        print(f"  Downloading 100s data block...")
        h1_full = TimeSeries.fetch_open_data('H1', t0, t0+101, sample_rate=fs).highpass(1000)
        l1_full = TimeSeries.fetch_open_data('L1', t0, t0+101, sample_rate=fs).highpass(1000)

        # 2. ESTABLISH NOISE BASELINE
        print(f"  Calibrating noise background...")
        h1_bg = TimeSeries.fetch_open_data('H1', t0-110, t0-10, sample_rate=fs).highpass(1000)
        l1_bg = TimeSeries.fetch_open_data('L1', t0-110, t0-10, sample_rate=fs).highpass(1000)
        
        slide_rs = []
        for i in range(200):
            shift = int((0.1 + i*0.05)*fs)
            r_bg, _ = pearsonr(np.roll(h1_bg.value[:fs], shift), l1_bg.value[:fs])
            slide_rs.append(r_bg)
        bg_mu, bg_std = np.mean(slide_rs), np.std(slide_rs)

        # 3. SCAN EVERY SECOND FOR 100 SECONDS
        print(f"  Scanning 100-second window...")
        flips = []
        for offset in range(0, 100):
            s1 = h1_full.value[offset*fs : (offset+1)*fs]
            s2 = l1_full.value[offset*fs : (offset+1)*fs]
            r_val, _ = pearsonr(s1, s2)
            flips.append((offset, r_val))
        
        best_off, flip_r = min(flips, key=lambda x: x[1])
        _, max_r = max(flips, key=lambda x: x[1])
        sigma_flip = (flip_r - bg_mu) / bg_std
        
        master_data.append([name, e_type, round(max_r, 4), round(flip_r, 4), round(sigma_flip, 2), best_off])
        print(f"  -> SUCCESS: Flip found at {best_off}s (r = {flip_r:.4f})")

        # 4. GENERATE AND SAVE IMAGE
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(h1_bg.psd(fftlength=1), label="Baseline", color='gray', alpha=0.5)
        h1_flip = h1_full.crop(t0+best_off, t0+best_off+1)
        ax.loglog(h1_flip.psd(fftlength=1), label=f"Flip (at {best_off}s)", color='purple', linewidth=2)
        
        f = np.logspace(3, 3.2, 100)
        ax.loglog(f, 1e-45 * (f**-1.024), 'k--', label='ABDM Floor')

        ax.set_title(f"{name}: Metric Lifecycle")
        ax.legend(loc='lower left')
        plt.savefig(f"{name}_Evidence.png", dpi=300)
        plt.close()

    except Exception as e:
        print(f"  Error processing {name}: {e}")
        continue

# SAVE FINAL MASTER TABLE
df = pd.DataFrame(master_data, columns=['ID', 'Type', 'Max_r', 'Flip_r', 'Sigma', 'Time_s'])
df.to_csv("Final_Master_Evidence_100s.csv", index=False)
print("\n--- ALL FIVE EVENTS COMPLETE ---")
