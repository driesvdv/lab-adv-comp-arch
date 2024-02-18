import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('timing_data.csv')

# Apply a moving average filter to smooth out the data
window_size = 5  # Adjust the window size as needed
df_smoothed = df.rolling(window=window_size).mean()

# Plot the smoothed data
plt.plot(df_smoothed['Array Size'], df_smoothed[' GPU Time (ms)'], label='GPU Time (Smoothed)')
plt.plot(df_smoothed['Array Size'], df_smoothed[' CPU Time (ms)'], label='CPU Time (Smoothed)')
plt.xlabel('Array Size')
plt.ylabel('Time (ms)')
plt.title('Smoothed GPU vs CPU Time')
#plt.xscale('log')
plt.legend()
plt.show()
