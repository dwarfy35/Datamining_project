def cluster_summary(df, cluster_col='cluster_label', feature_cols=None):
    if feature_cols is None:
        feature_cols = df.columns.difference([cluster_col, 'gameid', 'teamname', 'result'])
    
    cluster_profiles = df.groupby(cluster_col)[feature_cols].mean()
    overall_mean = df[feature_cols].mean()
    relative_profiles = cluster_profiles.div(overall_mean, axis=1)
    #relative_profiles = cluster_profiles # Don't div by the mean, cuz it is close to 0 for residuals
    
    return relative_profiles

import numpy as np
import matplotlib.pyplot as plt

def plot_outlier_distribution(df, column, title_suffix=""):
    """
    Plots a histogram comparing normal data and outliers for a specific column.
    
    Parameters:
    - df: The DataFrame containing the data.
    - column: The string name of the column to plot (e.g., 'kills_sup').
    - title_suffix: Optional string to add to the plot title.
    """
    data_min = df[column].min()
    data_max = df[column].max()
    
    num_bins = max(int(data_max - data_min + 1), 10)
    bin_edges = np.linspace(data_min, data_max, num_bins) 

    plt.figure(figsize=(10, 6))

    normal_data = df[df['is_outlier'] == 1][column]
    outlier_data = df[df['is_outlier'] == -1][column]

    plt.hist(normal_data, bins=bin_edges, alpha=0.4, label='Normal', color='steelblue')

    if len(outlier_data) > 0:
        scale_factor = len(normal_data) / len(outlier_data)
        weights = np.ones_like(outlier_data) * scale_factor
        plt.hist(outlier_data, bins=bin_edges, alpha=0.7, label='Outliers (Scaled)', 
                 weights=weights, color='peru')
    else:
        print(f"No outliers found in {column}.")

    plt.xlabel(column.replace('_', ' ').capitalize())
    plt.ylabel('Adjusted Frequency')
    plt.title(f'Distribution of {column} {title_suffix} (Outliers Highlighted)')
    plt.legend()
    plt.show()