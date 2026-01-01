import numpy as np
import pandas as pd

def calculate_psi(expected, actual, buckets=10):
    """
    Calculates PSI for a specific feature or probability score.
    expected: The distribution from training data
    actual: The distribution from current production data
    """
    def scale_range(input_data, buckets):
        # Create decile buckets based on the expected data
        breakpoints = np.percentile(input_data, np.arange(0, 101, 100 / buckets))
        # Ensure breakpoints are unique
        breakpoints = np.unique(breakpoints)
        return breakpoints

    breakpoints = scale_range(expected, buckets)
    
    # Categorize data into buckets
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Replace 0s with a tiny number to avoid division by zero
    expected_percents = np.clip(expected_percents, 0.0001, 1)
    actual_percents = np.clip(actual_percents, 0.0001, 1)

    # PSI Formula: (Actual% - Expected%) * ln(Actual% / Expected%)
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    total_psi = np.sum(psi_values)
    
    return total_psi

# --- Example Usage for GitHub ---
# train_incomes = [50k, 60k, 55k...] 
# production_incomes = [30k, 25k, 28k...] <- Market crash!
# print(f"Income PSI: {calculate_psi(train_incomes, production_incomes)}")