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


def plot_optics_reachability_and_results(optics_reachability, champs_2d_tsne, champs_cluster_labels, champs, eps=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.plot(optics_reachability)
    if eps is not None:
        ax1.axhline(y=eps, color='r', linestyle='--')
    ax1.set_title("OPTICS Reachability Plot")
    ax1.set_xlabel("Points in the order they were processed")
    ax1.set_ylabel("Reachability Distance")

    scatter = ax2.scatter(champs_2d_tsne[:, 0], champs_2d_tsne[:, 1], c=champs_cluster_labels)
    for i, champ in enumerate(champs.index):
        ax2.annotate(champ, (champs_2d_tsne[i, 0], champs_2d_tsne[i, 1]))
    ax2.set_title("t-SNE of Champions (OPTICS clusters)")
    plt.colorbar(scatter, ax=ax2, label="Cluster")

    n_clusters = len(set(champs_cluster_labels)) - (1 if -1 in champs_cluster_labels else 0)
    fig.suptitle(f"OPTICS Clustering: {n_clusters} clusters found", fontsize=14)

    plt.tight_layout()
    plt.show()

# Improved version which shows the clusters in the reachability plot as well for the Xi method, currently fails to count the clusters right and use consistent colors
# def plot_optics_reachability_and_results_v2(optics, optics_reachability, champs_2d_tsne, champs_cluster_labels, champs, eps=None):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

#     ax1.plot(optics_reachability, color='black', linewidth=0.8)

#     if eps is not None:
#         # DBSCAN method: show horizontal eps line
#         ax1.axhline(y=eps, color='r', linestyle='--', label=f'eps={eps}')
#         ax1.legend()
#     elif hasattr(optics, 'cluster_hierarchy_') and optics.cluster_hierarchy_ is not None:
#         # Xi method: shade each cluster's valley using cluster_hierarchy_
#         n_clusters = len(set(champs_cluster_labels)) - (1 if -1 in champs_cluster_labels else 0)
#         cmap = plt.get_cmap('tab10', n_clusters)
#         for cluster_id, (start, end) in enumerate(optics.cluster_hierarchy_[:, :2].astype(int)):
#             ax1.axvspan(start, end, alpha=0.3, color=cmap(cluster_id), label=f'Cluster {cluster_id}')
#         ax1.legend(fontsize=7, loc='upper right')

#     ax1.set_title("OPTICS Reachability Plot")
#     ax1.set_xlabel("Points in the order they were processed")
#     ax1.set_ylabel("Reachability Distance")

#     scatter = ax2.scatter(champs_2d_tsne[:, 0], champs_2d_tsne[:, 1], c=champs_cluster_labels)
#     for i, champ in enumerate(champs.index):
#         ax2.annotate(champ, (champs_2d_tsne[i, 0], champs_2d_tsne[i, 1]))
#     ax2.set_title("t-SNE of Champions (OPTICS clusters)")
#     plt.colorbar(scatter, ax=ax2, label="Cluster")

#     n_clusters = len(set(champs_cluster_labels)) - (1 if -1 in champs_cluster_labels else 0)
#     fig.suptitle(f"OPTICS Clustering: {n_clusters} clusters found", fontsize=14)

#     plt.tight_layout()
#     plt.show()