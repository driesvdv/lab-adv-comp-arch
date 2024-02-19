import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('timing_data.csv')

# Plot the data
plt.figure(figsize=(10, 6))

# CPU time
plt.plot(df['Array Size'], df['CPU Time (ms)'], label='CPU', marker='o')

# Atomic operation time
plt.plot(df['Array Size'], df['Atomic Time (ms)'], label='Atomic', marker='o')

# Reduction operation time
plt.plot(df['Array Size'], df['Reduction Time (ms)'], label='Reduction, iterative calls', marker='o')

# Set plot labels and title
plt.xlabel('Array Size')
plt.xscale('log', base=2)
plt.yscale('log')
plt.ylabel('Execution Time (ms)')
plt.title('Execution Time vs. Array Size')
plt.grid(True)
plt.legend()

# Show plot
plt.show()
