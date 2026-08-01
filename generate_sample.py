"""
Generate a larger sample SSD dataset for testing.
Run: python generate_sample.py
"""
import pandas as pd
import numpy as np

np.random.seed(42)
N = 500

data = {
    'wear_level':       np.random.uniform(5, 100, N).round(2),
    'pe_cycles':        np.random.randint(100, 4000, N),
    'temperature':      np.random.uniform(25, 70, N).round(1),
    'read_error_rate':  np.random.exponential(0.005, N).round(6),
    'write_error_rate': np.random.exponential(0.004, N).round(6),
    'bad_block_count':  np.random.randint(0, 60, N),
    'ecc_error_rate':   np.random.exponential(0.003, N).round(6),
    'power_on_hours':   np.random.randint(100, 50000, N),
}

df = pd.DataFrame(data)

# Synthesise failure label based on multiple factors
score = (df['wear_level']/100 * 0.4
       + df['pe_cycles']/4000 * 0.2
       + df['temperature']/70 * 0.2
       + df['read_error_rate']/df['read_error_rate'].max() * 0.1
       + df['bad_block_count']/60 * 0.1)

df['failure'] = (score > 0.65).astype(int)

df.to_csv('static/sample_ssd_data_large.csv', index=False)
print(f"Generated {N} device records. Failures: {df['failure'].sum()} ({df['failure'].mean()*100:.1f}%)")
