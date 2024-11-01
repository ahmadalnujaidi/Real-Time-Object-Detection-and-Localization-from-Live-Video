# FIRST RUN downloadModel.py to download the model
# SECOND RUN speedMeasurement.py to measure the SPEED
# THIRD RUN accuracyMeasurement.py to measure the ACCURACY
# FOURTH RUN memoryMeasurement.py to measure the MEMORY USAGE

# COMBINED SCORE!!!!!!

# measured values
S = 75.33  # Speed in ms (lower is better) CHANGE WITH SPEED MEASUREMENT
A = 1.0000  # Accuracy (e) (higher is better) CHANGE WITH ACCURACY MEASUREMENT
M = 14.07  # Memory usage in MB (lower is better) CHANGE WITH MEMORY MEASUREMENT

# Weights (w1, w2, w3) provided by the competition (example: equal weights)
w1, w2, w3 = 0.4, 0.4, 0.2  # Adjust these as per the competition rules

# Calculate the Combined Score (CS) directly using the given formula
CS = (w1 * S) + (w2 * A) + (w3 * M)

# Print the final Combined Score (CS)
print(f"Combined Score (CS): {CS:.4f}")

#------------------------------------------------------------------------------------------------------------
import pandas as pd

# Example: Save your metrics in results.csv OPTIONAL
results = pd.DataFrame({
    'Speed(ms)': [75.33], # enter speed
    'Accuracy(e)': [1.0000], # enter accuracy
    'Memory(MB)': [14.07], # enter memory
    'Combined Score': [33.3460] # final combined score
})

results.to_csv('results.csv', index=False)
print("results file generated: results.csv")
