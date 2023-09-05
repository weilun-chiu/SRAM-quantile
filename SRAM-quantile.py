import random
import numpy as np

cols = 1024   # assume each cache can store 1024 float
rows = 4096   # assume total data is cols * rows
num_bins = 8  # assume we wanto to quantize into int-8 (7 bins)

def normalize(data, disable = True):
  if disable: # Normalized can reduce the error but required additional memory to memorize mean & STDV
    return data 
  mean = np.mean(data)
  stdv = np.std(data)
  data = (np.array(data) - mean) / stdv
  return data

# Generate random floats between 0.0 and 100.0 for each element in the grid
random_grid = [[random.uniform(0.0, 100.0) for _ in range(cols)] for _ in range(rows)]

# Flatten the grid into a 1D array
flat_grid = [item for sublist in random_grid for item in sublist]
normalized_flat_grid = normalize(flat_grid)

# Calculate quantiles using numpy
quantiles = np.percentile(normalized_flat_grid, np.linspace(0, 100, num_bins + 1))
output_string = ", ".join([f'{x:.2f}' for x in quantiles])
# Print the quantiles
print(f"Full Matrix Quantile Results: {output_string}")

# Calculate quantiles for each sublist (row)
quantiles_per_row = []

for row in random_grid:
    normalized_row = normalize(row)
    quantiles = np.percentile(normalized_row, np.linspace(0, 100, num_bins + 1))
    quantiles_per_row.append(quantiles)

# Calculate the average quantiles
average_quantiles = np.mean(quantiles_per_row, axis=0)
output_string = ", ".join([f'{x:.2f}' for x in average_quantiles])
# Print the average quantiles
print(f"SRAM Quantile Results: {output_string}")

quantized_error = [abs(quantiles[i]-average_quantiles[i]) for i in range(num_bins+1)]
output_string = ", ".join([f'{x:.2f}' for x in quantized_error])
print(f"Quantile ERROR: {output_string}")
print(f'max = {max(quantized_error):.2f}, min = {min(quantized_error):.2f}, mean = {np.mean(quantized_error):.2f}, stdv = {np.std(quantized_error):.2f}')
