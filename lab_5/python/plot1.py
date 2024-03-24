import pandas as pd
import matplotlib.pyplot as plt

# Read data from file 1
data1 = pd.read_csv("timings/synchronous_cuda_results.csv")
x1 = data1["Array size"]
total_exec_time1 = data1[" Total Execution time (µs)"]

# Read data from file 2
data2 = pd.read_csv("timings/asynchronous_cuda_results.csv")
x2 = data2["Array Size"]
total_exec_time2 = data2["Execution time (µs)"]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x1, total_exec_time1, label="Synchronous", marker='o')
plt.plot(x2, total_exec_time2, label="Asynchronous", marker='o')

# Formatting
plt.xlabel("Array Size")
plt.ylabel("Total Execution Time (µs)")
plt.title("Total Execution Time vs. Array Size")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show plot
plt.show()
