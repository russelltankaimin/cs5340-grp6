import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # 1. Load the data
    file_path = 'results/music_metrics_flat.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)

    # 2. Identify the unique categories
    metrics = df['Metric'].unique()
    corruption_types = df['Corruption Type'].unique()

    # 3. Setup the figure
    # Create one subplot per metric vertically
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))

    # Ensure axes is iterable even if there's only 1 metric
    if len(metrics) == 1:
        axes = [axes]

    width = 0.35  # The width of the bars
    x = np.arange(len(corruption_types))  # The label locations

    # 4. Loop through each metric and plot
    for ax, metric in zip(axes, metrics):
        # Filter data for the current metric
        metric_df = df[df['Metric'] == metric]
        
        # Extract data for Corrupted and ensure it aligns with the x-axis order
        corrupted = metric_df[metric_df['Mode'] == 'corrupted'].set_index('Corruption Type').reindex(corruption_types)
        # Extract data for Reconstructed and ensure alignment
        reconstructed = metric_df[metric_df['Mode'] == 'reconstructed'].set_index('Corruption Type').reindex(corruption_types)
        
        # Plot Corrupted bars with SEM as error bars
        rects1 = ax.bar(x - width/2, corrupted['Mean'], width, 
                        yerr=corrupted['SEM'], capsize=5, 
                        label='Corrupted', color='#ff9999', edgecolor='black')
        
        # Plot Reconstructed bars with SEM as error bars
        rects2 = ax.bar(x + width/2, reconstructed['Mean'], width, 
                        yerr=reconstructed['SEM'], capsize=5, 
                        label='Reconstructed', color='#66b3ff', edgecolor='black')
        
        # Formatting
        ax.set_ylabel('Mean Value', fontsize=12)
        ax.set_title(f'Comparison of {metric.replace("_", " ").upper()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([ct.replace('_', ' ').title() for ct in corruption_types], fontsize=11)
        
        # Move legend outside the plot area to avoid covering data
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add the text annotations on top of the bars
        ax.bar_label(rects1, padding=5, fmt='%.2f', fontsize=10)
        ax.bar_label(rects2, padding=5, fmt='%.2f', fontsize=10)
        
        # Add a subtle grid for easier reading
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Extend the y-limit slightly so the text labels don't get cut off at the top
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)

    plt.tight_layout()
    output_path = 'results/metrics_comparison_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
