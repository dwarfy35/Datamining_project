def cluster_summary(df, cluster_col='cluster_label', feature_cols=None):
    if feature_cols is None:
        feature_cols = df.columns.difference([cluster_col, 'gameid', 'teamname', 'result'])
    
    cluster_profiles = df.groupby(cluster_col)[feature_cols].mean()
    overall_mean = df[feature_cols].mean()
    relative_profiles = cluster_profiles.div(overall_mean, axis=1)
    #relative_profiles = cluster_profiles # Don't div by the mean, cuz it is close to 0 for residuals
    
    return relative_profiles