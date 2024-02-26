import matplotlib.pyplot as plt

# Data
warps = []
block_sizes = []
execution_times = []

# Read data from the provided list
with open('timing_data.csv', 'r') as file:
    next(file)  # Skip header
    for line in file:
        warp, block_size, exec_time = line.strip().split(', ')
        warps.append(int(warp))
        block_sizes.append(int(block_size))
        execution_times.append(float(exec_time))

# Plot
#plt.figure(figsize=(10, 6))
#plt.plot(block_sizes,warps, linestyle='-', marker='o', color='b')
#plt.title('Warps vs Execution Time averaged out over 1000 runs')
#plt.xlabel('Block size')
#plt.ylabel('Warps')
#plt.yscale('log')
#x_ticks = range(0, max(block_sizes) + 1, 32)
#plt.xticks(x_ticks)
#plt.grid(True)
#plt.show()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(block_sizes, execution_times, linestyle='-', marker='o', color='b')
plt.title('Warps vs Execution Time averaged out over 1000 runs')
plt.xlabel('Execution Time (ms)')
plt.ylabel('Block sizes')
plt.yscale('log')
#x_ticks = range(0, max(block_sizes) + 1, 32)
#plt.xticks(x_ticks)
plt.grid(True)
plt.show()

