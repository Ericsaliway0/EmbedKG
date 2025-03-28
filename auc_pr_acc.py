import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Models, metrics, and datasets
models = ['EGCN', 'GCN', 'GAT', 'GraphSAGE', 'GIN', 'ChebNet']
metrics = ['Accuracy', 'AUROC', 'AUPR']
datasets = ['TarBase', 'miRTarBase', 'miRNet']

# Data with mean ± std values
data_raw = {
    'Dataset': ['TarBase'] * 3 + ['miRTarBase'] * 3 + ['miRNet'] * 3,
    'Metric': ['Accuracy', 'AUROC', 'AUPR'] * 3,
    'EGCN': ["0.9368 ± 0.0090", "0.9722 ± 0.0094", "0.9558 ± 0.0087", 
                "0.9261 ± 0.0062", "0.9626 ± 0.0049", "0.9463 ± 0.0079", 
                "0.9980 ± 0.0080", "0.9950 ± 0.0060", "0.9953 ± 0.0105"],
    'GCN': ["0.9285 ± 0.0097", "0.9883 ± 0.0109", "0.9746 ± 0.0099", 
            "0.9238 ± 0.0176", "0.9602 ± 0.0122", "0.9481 ± 0.0182", 
            "0.9487 ± 0.0162", "0.9950 ± 0.0087", "0.9891 ± 0.0147"],
    'GAT': ["0.9297 ± 0.0108", "0.9492 ± 0.0097", "0.9255 ± 0.0156", 
            "0.8978 ± 0.0181", "0.9276 ± 0.0103", "0.8818 ± 0.0099", 
            "0.9985 ± 0.0117", "0.9949 ± 0.0211", "0.9952 ± 0.0164"],
    'GraphSAGE': ["0.8937 ± 0.0079", "0.9511 ± 0.0139", "0.9181 ± 0.0166", 
                  "0.9408 ± 0.0181", "0.9452 ± 0.0099", "0.9215 ± 0.0099", 
                  "0.9387 ± 0.0143", "0.9952 ± 0.0111", "0.9962 ± 0.0172"],
    'GIN': ["0.9345 ± 0.0110", "0.9713 ± 0.0123", "0.9540 ± 0.0115", 
            "0.8451 ± 0.0161", "0.9358 ± 0.0125", "0.9159 ± 0.0108", 
            "0.9293 ± 0.0131", "0.9675 ± 0.0134", "0.9505 ± 0.0123"],
    'ChebNet': ["0.9406 ± 0.0136", "0.9715 ± 0.0149", "0.9575 ± 0.0087", 
                "0.8983 ± 0.0103", "0.9476 ± 0.0156", "0.9256 ± 0.0095", 
                "0.9060 ± 0.0214", "0.9270 ± 0.0089", "0.8675 ± 0.0205"]
}

# Convert to DataFrame
df = pd.DataFrame(data_raw)

# Function to split mean and std
def split_mean_std(value):
    if '±' in value:
        mean, std = value.split(' ± ')
        return float(mean), float(std)
    return float(value), 0.0  # Default std = 0 if missing

# Process the DataFrame to extract mean and std
for model in models:
    df[[f'{model}_mean', f'{model}_std']] = df[model].apply(lambda x: pd.Series(split_mean_std(x)))

# Melt data for Seaborn (long format)
df_melted = df.melt(
    id_vars=['Dataset', 'Metric'],
    value_vars=[f"{model}_mean" for model in models],
    var_name='Model',
    value_name='Score'
)

# Extract model name from 'Model' column
df_melted['Model'] = df_melted['Model'].str.replace('_mean', '')

# Extract standard deviations separately
df_std = df.melt(
    id_vars=['Dataset', 'Metric'],
    value_vars=[f"{model}_std" for model in models],
    var_name='Model',
    value_name='Error'
)
df_std['Model'] = df_std['Model'].str.replace('_std', '')

# Merge the data to get both mean and std in one DataFrame
df_melted = df_melted.merge(df_std, on=['Dataset', 'Metric', 'Model'])

# Set seaborn style
##sns.set(style="whitegrid")
palette = sns.color_palette("husl", len(models))

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey='row')

# Plot each metric-dataset combination
for i, metric in enumerate(metrics):
    for j, dataset in enumerate(datasets):
        ax = axes[i, j]
        subset = df_melted[(df_melted['Metric'] == metric) & (df_melted['Dataset'] == dataset)]

        # Ensure yerr matches y's shape
        yerr_values = subset['Error'].values if len(subset) > 1 else None

        sns.barplot(
            x='Model', y='Score', hue='Model',
            data=subset, palette=palette,
            ax=ax, capsize=0.1, errcolor='black', errwidth=1.5,
            errorbar=None  # Disable Seaborn’s internal error bar calculation
        )

        # Add manual error bars using matplotlib
        for k, bar in enumerate(ax.patches):
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            if yerr_values is not None and k < len(yerr_values):
                ax.errorbar(x, height, yerr=yerr_values[k], fmt='none', color='black', capsize=3)

        # Set title only for the first row
        if i == 0:
            ax.set_title(f"{dataset}", fontsize=14, pad=20)
        else:
            ax.set_title("")

        # Remove x-tick labels and ticks
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel('')


        # Set y-axis labels only for the first column
        if j == 0:
            ax.set_ylabel(metric, labelpad=20, fontsize=14)
        else:
            ax.set_ylabel('')

        # Remove subplot legends safely
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # Add black border around each subplot
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

        # Set y-axis limits
        ax.set_ylim(0.6, 1.05)

# Adjust layout to make room for the legend and tighten the spacing
fig.subplots_adjust(top=0.92, bottom=0.12)

# Bring the legend closer to the plots
handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(len(models))]
fig.legend(
    handles, models, loc='lower center',
    bbox_to_anchor=(0.5, 0.03), ncol=len(models),
    fontsize=12, frameon=False
)
plt.savefig('model_performance_comparison.png', bbox_inches='tight', dpi=300)

plt.show()
