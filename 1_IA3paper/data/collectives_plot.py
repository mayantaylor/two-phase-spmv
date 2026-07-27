import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('collectives_scaleB.csv')

# Plot
plt.figure(figsize=(10, 6))

# also plot model:
# reduce model: vector_length + 12 * (512 + 1) + 9/64 * 512
df['reduce-model'] = df['vector length'] + 12 * (512 + 1) + 8 * 9
df['bcast-model'] =  df['vector length'] + 2 * (512 + 2) + 8 * 9
df['old-reduce-model'] = df['vector length'] + 12 * (512 + 1) 
df['old-bcast-model'] =  df['vector length'] +  (512 + 2) 

plt.plot(df['vector length'], df['reduce'] / 1100, marker='o', linewidth=2, label='Reduce', color='red')
plt.plot(df['vector length'], df['reduce-model'] / 1100, linewidth=2, label='Reduce Model', color='red', linestyle='--')
plt.plot(df['vector length'], df['old-reduce-model'] / 1100, linewidth=2, label='Luczynski Reduce Model', color='black', linestyle='--')

plt.plot(df['vector length'], df['bcast-tomem'] / 1100, marker='o', linewidth=2, label='Broadcast', color='blue')
plt.plot(df['vector length'], df['bcast-model'] / 1100, linewidth=2, label='Broadcast Model', color='blue', linestyle='--')
plt.plot(df['vector length'], df['old-bcast-model'] / 1100, linewidth=2, label='Luczynski Broadcast Model', color='black', linestyle='--')


plt.xlabel('Vector Length')
plt.ylabel('Time (clock cycles)')
plt.title('1D Collectives Scaling B')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.xscale('log', base=2)
plt.legend()

plt.savefig('../figures/collectives_scaleB.png', dpi=300)


# MAKE FIGURE 2
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

plt.plot(df['nPEs'], df['reduce'] / 1100 , marker='o', linewidth=2, label='Reduce', color='red')
plt.plot(df['nPEs'], df['reduce-model'] / 1100, linewidth=2, label='Reduce Model', color='red', linestyle='--')
plt.plot(df['nPEs'], df['old-reduce-model'] / 1100, linewidth=2, label='Luczynski Reduce Model', color='black', linestyle='--')

plt.plot(df['nPEs'], df['bcast-tomem'] / 1100, marker='o', linewidth=2, label='Broadcast', color='blue')
plt.plot(df['nPEs'], df['bcast-model'] / 1100, linewidth=2, label='Broadcast Model', color='blue', linestyle='--')
plt.plot(df['nPEs'], df['old-bcast-model']  / 1100, linewidth=2, label='Luczynski Bcast Model', color='black', linestyle='--')


plt.xlabel('nPEs')
plt.ylabel('Time (clock cycles)')
plt.title('1D Collectives Scaling P')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.xscale('log', base=2)
plt.legend()

plt.savefig('../figures/collectives_scaleP.png', dpi=300)
