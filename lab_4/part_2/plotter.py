import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the CSV file
folder_path = os.path.join(current_dir, "timings")

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# Plot execution times for each CSV file
for file_name in csv_files:
    print(f"Plotting {file_name}...")
    # Read the CSV file
    csv_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(csv_path)
    
    print(data.head())
    
    # Extract matrix sizes and execution times
    matrix_sizes = data['Matrix_Size']
    execution_times = data['Execution_Time(us)']
    
    # Plot execution times
    plt.plot(matrix_sizes, execution_times, label=file_name)

# Add labels and title
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (ms)')
plt.title('Execution Time vs. Matrix Size')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
