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

plt.plot(df['vector length'], df['reduce'] / 1000, marker='o', linewidth=2, label='Reduce', color='red')
plt.plot(df['vector length'], df['reduce-model'] / 1000, linewidth=2, label='Reduce Model', color='red', linestyle='--')

plt.plot(df['vector length'], df['bcast-tomem'] / 1000, marker='o', linewidth=2, label='Broadcast', color='blue')
plt.plot(df['vector length'], df['bcast-model'] / 1000, linewidth=2, label='Broadcast Model', color='blue', linestyle='--')
plt.plot(df['vector length'], df['old-bcast-model'] / 1000, linewidth=2, label='Luczynski Broadcast Model', color='black', linestyle='--')


plt.xlabel('B', fontsize=20)
plt.ylabel('Time (µs)', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.xscale('log', base=2)

plt.savefig('../figures/collectives_scaleB.pdf', dpi=300, format='pdf')
