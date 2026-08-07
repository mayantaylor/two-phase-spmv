import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from style import figure, style_axes, save


# Load the CSV file
df = pd.read_csv('collectives_scaleB.csv')

fig, ax = figure("rect")

# also plot model:
# reduce model: vector_length + 12 * (512 + 1) + 9/64 * 512
df['reduce-model'] = df['vector length'] + 12 * (512 + 1) + 8 * 9
df['bcast-model'] =  df['vector length'] + 2 * (512 + 2) + 8 * 9
df['old-reduce-model'] = df['vector length'] + 12 * (512 + 1) 
df['old-bcast-model'] =  df['vector length'] +  (512 + 2) 

plt.plot(df['vector length'], df['reduce'] / 1000, marker='o',  label='Reduce', color='red')
plt.plot(df['vector length'], df['reduce-model'] / 1000,  label='Reduce Model', color='red', linestyle='--')

plt.plot(df['vector length'], df['bcast-tomem'] / 1000, marker='o',  label='Broadcast', color='blue')
plt.plot(df['vector length'], df['bcast-model'] / 1000,  label='Broadcast Model', color='blue', linestyle='--')
plt.plot(df['vector length'], df['old-bcast-model'] / 1000,  label='Luczynski Broadcast Model', color='black', linestyle='--')


plt.xlabel('B')
plt.ylabel('Time (µs)')

plt.xscale('log', base=2)
style_axes(ax)
save(fig,'../figures/collectives_scaleB.pdf')
