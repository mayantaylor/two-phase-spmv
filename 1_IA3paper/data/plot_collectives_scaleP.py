import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('collectives_scaleP.csv')

# Plot
plt.figure(figsize=(10, 6))

# also plot model:
# reduce model: vector_length + 12 * (512 + 1) + 9/64 * 512
df['reticle-overhead'] = np.floor((df['nPEs'] + 2) / 64) * 9
df['reduce-model'] = 256 + 12 * (df['nPEs'] + 1) + df['reticle-overhead']
df['bcast-model'] =  256 + 2 * (df['nPEs'] + 2) + df['reticle-overhead']
df['old-reduce-model'] = 256 + 12 * (df['nPEs'] + 1)
df['old-bcast-model'] =  256 + (df['nPEs'] + 2) 

plt.plot(df['nPEs'], df['reduce'] / 1000 , marker='o', linewidth=2, label='Reduce', color='red')
plt.plot(df['nPEs'], df['reduce-model'] / 1000, linewidth=2, label='Reduce Model', color='red', linestyle='--')

plt.plot(df['nPEs'], df['bcast-tomem'] / 1000, marker='o', linewidth=2, label='Broadcast', color='blue')
plt.plot(df['nPEs'], df['bcast-model'] / 1000, linewidth=2, label='Broadcast Model', color='blue', linestyle='--')
plt.plot(df['nPEs'], df['old-bcast-model']  / 1000, linewidth=2, label='Luczynski Broadcast Model', color='black', linestyle='--')

plt.xlabel('P', fontsize=20)
plt.ylabel('Time (µs)', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.xscale('log', base=2)
plt.legend(fontsize=20)

plt.savefig('../figures/collectives_scaleP.pdf', dpi=300, format='pdf')
