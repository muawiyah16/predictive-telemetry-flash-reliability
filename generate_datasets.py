"""
SSD Failure Analysis System – Large Dataset Generator
Creates 10 diverse SSD CSV datasets for testing.
Run: python generate_datasets.py
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = os.path.join('static', 'datasets')
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(0)

# ─── Dataset profiles ────────────────────────────────────────────────────────
PROFILES = [
    {
        'filename': 'dataset_01_enterprise_healthy.csv',
        'title':    'Enterprise Data Center – Healthy Fleet',
        'n':        1000,
        'wear_range':    (5,  55),
        'pe_range':      (200, 1800),
        'temp_range':    (25, 45),
        'err_scale':     0.001,
        'bb_max':        10,
        'failure_bias':  0.08,   # low failure rate
        'profile':       'new',
    },
    {
        'filename': 'dataset_02_aging_servers.csv',
        'title':    'Aging Server Farm – Mixed Health',
        'n':        1200,
        'wear_range':    (30, 85),
        'pe_range':      (800, 3200),
        'temp_range':    (35, 62),
        'err_scale':     0.008,
        'bb_max':        35,
        'failure_bias':  0.28,
        'profile':       'mixed',
    },
    {
        'filename': 'dataset_03_critical_datacenter.csv',
        'title':    'Critical Data Center – High Risk Fleet',
        'n':        800,
        'wear_range':    (60, 99),
        'pe_range':      (2000, 4000),
        'temp_range':    (50, 72),
        'err_scale':     0.025,
        'bb_max':        70,
        'failure_bias':  0.55,   # high failure rate
        'profile':       'old',
    },
    {
        'filename': 'dataset_04_cloud_storage.csv',
        'title':    'Cloud Storage Provider – Large Scale',
        'n':        1500,
        'wear_range':    (10, 70),
        'pe_range':      (300, 2800),
        'temp_range':    (28, 55),
        'err_scale':     0.004,
        'bb_max':        25,
        'failure_bias':  0.18,
        'profile':       'mixed',
    },
    {
        'filename': 'dataset_05_edge_computing.csv',
        'title':    'Edge Computing Nodes – Temperature Stressed',
        'n':        700,
        'wear_range':    (20, 80),
        'pe_range':      (500, 2500),
        'temp_range':    (45, 78),    # hot environment
        'err_scale':     0.012,
        'bb_max':        45,
        'failure_bias':  0.35,
        'profile':       'hot',
    },
    {
        'filename': 'dataset_06_hpc_cluster.csv',
        'title':    'HPC Cluster – High Write Workload',
        'n':        1100,
        'wear_range':    (40, 95),
        'pe_range':      (1500, 4500),  # very high PE cycles
        'temp_range':    (30, 58),
        'err_scale':     0.015,
        'bb_max':        55,
        'failure_bias':  0.40,
        'profile':       'write_heavy',
    },
    {
        'filename': 'dataset_07_backup_storage.csv',
        'title':    'Backup Storage – Light Workload',
        'n':        900,
        'wear_range':    (3,  40),
        'pe_range':      (50, 800),
        'temp_range':    (22, 38),
        'err_scale':     0.0005,
        'bb_max':        5,
        'failure_bias':  0.05,
        'profile':       'light',
    },
    {
        'filename': 'dataset_08_mixed_fleet_2024.csv',
        'title':    'Mixed Fleet 2024 – Multi-Vendor',
        'n':        1300,
        'wear_range':    (15, 92),
        'pe_range':      (400, 3800),
        'temp_range':    (27, 65),
        'err_scale':     0.010,
        'bb_max':        50,
        'failure_bias':  0.30,
        'profile':       'mixed',
    },
    {
        'filename': 'dataset_09_nvme_fleet.csv',
        'title':    'NVMe SSD Fleet – High-Performance',
        'n':        850,
        'wear_range':    (8,  60),
        'pe_range':      (300, 2200),
        'temp_range':    (30, 52),
        'err_scale':     0.002,
        'bb_max':        15,
        'failure_bias':  0.12,
        'profile':       'nvme',
    },
    {
        'filename': 'dataset_10_qlc_endurance.csv',
        'title':    'QLC NAND – Low Endurance Devices',
        'n':        1000,
        'wear_range':    (25, 99),
        'pe_range':      (100, 1500),   # QLC has low PE cycles
        'temp_range':    (32, 65),
        'err_scale':     0.020,
        'bb_max':        80,            # more bad blocks
        'failure_bias':  0.48,
        'profile':       'qlc',
    },
]

# ─── Generator ────────────────────────────────────────────────────────────────
def make_dataset(p, seed):
    np.random.seed(seed)
    n = p['n']
    wl = p['wear_range']
    pe = p['pe_range']
    tp = p['temp_range']
    es = p['err_scale']
    bb = p['bb_max']

    wear_level       = np.random.uniform(wl[0], wl[1], n).round(2)
    pe_cycles        = np.random.randint(pe[0], pe[1] + 1, n)
    temperature      = np.random.uniform(tp[0], tp[1], n).round(1)
    read_error_rate  = np.abs(np.random.normal(es * 0.8, es * 0.4, n)).round(6)
    write_error_rate = np.abs(np.random.normal(es * 0.6, es * 0.3, n)).round(6)
    bad_block_count  = np.random.randint(0, bb + 1, n)
    ecc_error_rate   = np.abs(np.random.normal(es * 0.4, es * 0.2, n)).round(6)
    power_on_hours   = np.random.randint(100, 52000, n)

    # Realistic write amplification factor (1.0 – 5.0)
    write_amplification = np.random.uniform(1.0, 5.0, n).round(3)

    # Reallocated sectors (SMART attr 5)
    reallocated_sectors = np.random.randint(0, max(1, bb // 2), n)

    # Uncorrectable errors count
    uncorrectable_errors = np.random.randint(0, max(1, int(es * 10000)), n)

    # Power cycle count
    power_cycle_count = np.random.randint(10, 5000, n)

    # Compute a weighted failure score
    score = (
        (wear_level / 100)       * 0.35
      + (pe_cycles  / pe[1])     * 0.20
      + (temperature / tp[1])    * 0.15
      + (read_error_rate / (es * 2 + 1e-9)) * 0.10
      + (bad_block_count / (bb + 1)) * 0.12
      + (ecc_error_rate  / (es + 1e-9)) * 0.08
    )
    score = np.clip(score, 0, 1)

    # Threshold tuned so ~failure_bias fraction become 1
    threshold = np.quantile(score, 1 - p['failure_bias'])
    failure = (score > threshold).astype(int)

    df = pd.DataFrame({
        'device_id':            [f"SSD-{p['profile'].upper()}-{i+1:04d}" for i in range(n)],
        'wear_level':           wear_level,
        'pe_cycles':            pe_cycles,
        'temperature':          temperature,
        'read_error_rate':      read_error_rate,
        'write_error_rate':     write_error_rate,
        'bad_block_count':      bad_block_count,
        'ecc_error_rate':       ecc_error_rate,
        'power_on_hours':       power_on_hours,
        'write_amplification':  write_amplification,
        'reallocated_sectors':  reallocated_sectors,
        'uncorrectable_errors': uncorrectable_errors,
        'power_cycle_count':    power_cycle_count,
        'failure':              failure,
    })
    return df


# ─── Generate & save ─────────────────────────────────────────────────────────
print("=" * 60)
print("  SSD Dataset Generator – Creating 10 Dataset Files")
print("=" * 60)

summary_rows = []
for idx, profile in enumerate(PROFILES, start=1):
    df = make_dataset(profile, seed=idx * 7)
    out_path = os.path.join(OUTPUT_DIR, profile['filename'])
    df.to_csv(out_path, index=False)
    failures = int(df['failure'].sum())
    pct = failures / len(df) * 100
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  [{idx:02d}] {profile['filename']}")
    print(f"       Rows: {len(df):,}  |  Failures: {failures} ({pct:.1f}%)  |  Size: {size_kb:.1f} KB")
    summary_rows.append({
        'File': profile['filename'],
        'Title': profile['title'],
        'Rows': len(df),
        'Failure Count': failures,
        'Failure Rate': f"{pct:.1f}%",
        'Size (KB)': round(size_kb, 1),
    })

print()
print("=" * 60)
print(f"  All 10 datasets saved to: {os.path.abspath(OUTPUT_DIR)}")
print("=" * 60)

# Save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, '_dataset_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"\n  Summary saved to: {summary_path}")
print()
print(summary_df.to_string(index=False))
