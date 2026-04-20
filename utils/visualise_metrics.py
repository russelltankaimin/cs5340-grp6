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

    # Dictionary for directional annotations
    metric_directions = {
        'chroma_cos_sim': 'Higher is better ↑',
        'mfcc_dist': 'Lower is better ↓',
        'peaq_nmr_db': 'Lower is better ↓',
        'fad': 'Lower is better ↓'
    }

    # 3. Setup the figure in a 2x2 grid
    fig, axes_flat = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes_flat.flatten()

    width = 0.35  # The width of the bars
    x = np.arange(len(corruption_types))  # The label locations

    # 4. Loop through each metric and plot
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Filter data for the current metric
        metric_df = df[df['Metric'] == metric]
        
        # Extract data for Corrupted and ensure it aligns with the x-axis order
        corrupted = metric_df[metric_df['Mode'] == 'corrupted'].set_index('Corruption Type').reindex(corruption_types)
        # Extract data for Reconstructed and ensure alignment
        reconstructed = metric_df[metric_df['Mode'] == 'reconstructed'].set_index('Corruption Type').reindex(corruption_types)
        
        # Plot Corrupted bars with SEM as error bars
        rects1 = ax.bar(x - width/2, corrupted['Mean'], width, 
                        yerr=corrupted['SEM'], error_kw={'elinewidth': 2, 'capthick': 2, 'capsize': 5},
                        label='Corrupted', color='#ff9999', edgecolor='black')
        
        # Plot Reconstructed bars with SEM as error bars
        rects2 = ax.bar(x + width/2, reconstructed['Mean'], width, 
                        yerr=reconstructed['SEM'], error_kw={'elinewidth': 2, 'capthick': 2, 'capsize': 5},
                        label='Reconstructed', color='#66b3ff', edgecolor='black')
        
        # Formatting
        ax.set_ylabel('Mean Value', fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").upper()}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([ct.replace('_', ' ').title() for ct in corruption_types], fontsize=10)
        
        # Place legend INSIDE the graph (upper right)
        ax.legend(loc='upper right', framealpha=0.8)

        # Move arrow/insight to TOP LEFT inside the graph
        direction_text = metric_directions.get(metric, "")
        if direction_text:
            ax.annotate(direction_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                        fontsize=11, fontweight='bold', style='italic', va='top', ha='left',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Add the text annotations on top of the bars with ± values
        labels1 = [f"{m:.1f} \u00B1 {e:.2f}" for m, e in zip(corrupted['Mean'], corrupted['SEM'])]
        labels2 = [f"{m:.1f} \u00B1 {e:.2f}" for m, e in zip(reconstructed['Mean'], reconstructed['SEM'])]
        
        ax.bar_label(rects1, labels=labels1, padding=5, fontsize=10)
        ax.bar_label(rects2, labels=labels2, padding=5, fontsize=10)
        
        # Add a subtle grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # PEAQ NMR specific annotations
        if metric == 'peaq_nmr_db':
            ax.axhline(0, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
            # Add annotation for imperceptible values top-left, below direction text
            ax.annotate('Values < 0: Imperceptible', 
                        xy=(0.02, 0.88), 
                        xycoords='axes fraction',
                        color='red', fontsize=10, ha='left', va='top', style='italic',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        # Adjust ylim: Leave 25% extra space at top for labels and legends
        ymin, ymax = ax.get_ylim()
        range_val = ymax - ymin
        ax.set_ylim(ymin, ymax + (range_val * 0.25))

    # Remove empty subplots if there are fewer than 4 metrics
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    output_path = 'results/metrics_comparison_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
