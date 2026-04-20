import json
import pandas as pd
import os

def reorganize_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    corruption_types = data.get('corruption_types', {})
    
    for corruption, modes in corruption_types.items():
        for mode, metrics in modes.items():
            for metric_name, stats in metrics.items():
                row = {
                    'Corruption Type': corruption,
                    'Mode': mode,
                    'Metric': metric_name,
                    'Mean': stats.get('mean'),
                    'Std': stats.get('std'),
                    'SEM': stats.get('sem'),
                    'N': stats.get('n')
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    json_path = 'results/music_metrics_report.json'
    if os.path.exists(json_path):
        df = reorganize_json(json_path)
        
        # Save to CSV for the user to download/view
        output_csv = 'results/music_metrics_flat.csv'
        df.to_csv(output_csv, index=False)
        print(f"Flattened data saved to {output_csv}")
        
        # Display a pivot-table like view
        pivot_df = df.pivot_table(
            index=['Metric', 'Corruption Type'], 
            columns='Mode', 
            values='Mean'
        )
        print("\nPivot Table View (Mean Values):")
        print(pivot_df)
    else:
        print(f"File {json_path} not found.")
