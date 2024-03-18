import os
import matplotlib.pyplot as plt
import pandas as pd

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the CSV file
csv_file = os.path.join(current_dir, "timing_data_comparison.csv")

# Read the data from the CSV file
data = pd.read_csv(csv_file)

# Separate coalesced and non-coalesced data
coalesced_data = data[data["Method"] == "Coalesced"]
non_coalesced_data = data[data["Method"] == "Non-coalesced"]

# Plot coalesced data with rolling average
plt.plot(coalesced_data[' Block Size'], coalesced_data[' Average Execution Time (µs)'],
         marker='o', label='Coalesced', alpha=0.5)  # Set alpha to make the plot semi-transparent
plt.plot(coalesced_data[' Block Size'], coalesced_data[' Average Execution Time (µs)'].rolling(window=30, min_periods=1).mean(),
         color='blue', label='Coalesced (Rolling Avg)')

# Plot non-coalesced data with rolling average
plt.plot(non_coalesced_data[' Block Size'], non_coalesced_data[' Average Execution Time (µs)'],
         marker='x', label='Non-coalesced', alpha=0.5)  # Set alpha to make the plot semi-transparent
plt.plot(non_coalesced_data[' Block Size'], non_coalesced_data[' Average Execution Time (µs)'].rolling(window=30, min_periods=1).mean(),
         color='red', label='Non-coalesced (Rolling Avg)')

# Add labels and title
plt.xlabel('Block Size')
plt.ylabel('Average Execution Time (µs)')
plt.title('Comparison of Coalesced and Non-coalesced Memory Access with Rolling Average')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
