import numpy as np
from scipy import stats

# Example performance data for two models
model1_results = np.array([0.95, 0.94, 0.96, 0.97, 0.93])  # Model 1 (e.g., EGCN)
model2_results = np.array([0.87, 0.88, 0.85, 0.86, 0.84])  # Model 2 (e.g., EGCN-MLP)

# Perform t-test
t_statistic, p_value = stats.ttest_ind(model1_results, model2_results)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")
