import json
from typing import Optional

import pandas as pd


def create_metrics_table(metrics_json_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convert metrics JSON into a readable table format.
    
    Args:
        metrics_json_path (str): Path to the metrics JSON file
        output_path (Optional[str]): Path to save the table (CSV format). If None, won't save
        
    Returns:
        pd.DataFrame: DataFrame containing the metrics table
    """
    # Read the JSON file
    with open(metrics_json_path, 'r') as f:
        metrics = json.load(f)
    
    # Initialize list to store rows
    rows = []
    
    # Process each task
    for task_name, task_metrics in metrics.items():
        if task_name == 'idioms':
            # Handle idioms which has nested metrics
            for sub_task, sub_metrics in task_metrics.items():
                row = {
                    'Task': f'idioms_{sub_task}',
                    'Accuracy': sub_metrics['accuracy'],
                    'Precision': sub_metrics['precision'],
                    'Recall': sub_metrics['recall'],
                    'F1': sub_metrics['f1']
                }
                rows.append(row)
        else:
            # Handle other tasks
            row = {
                'Task': task_name,
                'Accuracy': task_metrics['accuracy'],
                'Precision': task_metrics['precision'],
                'Recall': task_metrics['recall'],
                'F1': task_metrics['f1']
            }
            # Add ROUGE metrics if they exist
            if 'rouge1_f' in task_metrics:
                row.update({
                    'ROUGE-1 F1': task_metrics['rouge1_f'],
                    'ROUGE-2 F1': task_metrics['rouge2_f'],
                    'ROUGE-L F1': task_metrics['rougeL_f']
                })
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Round numeric columns to 3 decimal places
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    
    # Save to CSV if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[INFO] Metrics table saved to: {output_path}")
    
    return df
