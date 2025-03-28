from scipy.stats import wilcoxon
import numpy as np

# Example performance data for two models (e.g., AUPRC or ACC)
model1_results = np.array([0.95, 0.94, 0.96, 0.97, 0.93])
model2_results = np.array([0.87, 0.88, 0.85, 0.86, 0.84])

# Perform Wilcoxon Signed-Rank Test
stat, p_value = wilcoxon(model1_results, model2_results)

print(f"Wilcoxon statistic: {stat}")
print(f"P-value: {p_value}")
