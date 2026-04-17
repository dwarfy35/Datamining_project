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


def plot_adj_matrix_sorted_by_clustering(adj_matrix, clustering, ax=None):
    if ax is None:
        ax = plt.gca()

    num_clusters = clustering.max() + 1
    #print(num_clusters)

    sorted_indicies = []

    cluster_delimiters = [0]

    for cluster_id in range(num_clusters):
        cluster_indices = np.where(clustering == cluster_id)[0]
        sorted_indicies.extend(cluster_indices)
        cluster_delimiters.append(len(sorted_indicies))
        
        #print(f"Cluster {cluster_id}: {len(cluster_indices)} players")

    sorted_adj_matrix = adj_matrix[sorted_indicies, :][:, sorted_indicies]
    ax.matshow(sorted_adj_matrix)

    #for delimiter in cluster_delimiters:
    #    ax.plot([0, len(sorted_adj_matrix)], [delimiter, delimiter], color="red")
    #    ax.plot([delimiter, delimiter], [0, len(sorted_adj_matrix)], color="red")

    #Instead of full lines, just box the clusters in the adj matrix
    for i in range(num_clusters):
        start = cluster_delimiters[i]
        end = cluster_delimiters[i+1]
        ax.plot([start, end-1, end-1, start, start], [start, start, end-1, end-1, start], color="red")

    # Rendering now handled outside the function
    #plt.show()

from .constants import league_to_region_dict


def count_cluster_regions(cluster_labels, uniq_player_ids, data):
    num_clusters = cluster_labels.max() + 1
    for i in range(num_clusters):
        region_count = dict.fromkeys(set(league_to_region_dict.values()), 0)
        for playerid in uniq_player_ids[cluster_labels == i]:
            leagues = data[data["playerid"] == playerid]["league"].unique()
            for league in leagues:
                region = league_to_region_dict[league]
                region_count[region] += 1
        print(i)
        print(region_count)
        print("__")