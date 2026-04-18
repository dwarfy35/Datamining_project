import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from utils.constants import league_to_region_dict
from utils.constants import league_to_continent_dict

# calculate the normalized mutual information score for the predicted labels compared to to players labelled by the league theyve played the most in 
def score_leagues(labels_pred, uniq_player_ids, data, score_fn):
    player_most_played_league = data.groupby("playerid")["league"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    assert not player_most_played_league.isna().any()


    #print("Total number of leagues:", len(player_most_played_league.unique()))

    # make sure the order of the true labels are the same as the order of the predictions
    uniq_player_ids_most_played_league = np.vectorize(lambda x: player_most_played_league[x])(uniq_player_ids) 
    _, league_labels = np.unique(uniq_player_ids_most_played_league, return_inverse=True)

    score = score_fn(league_labels, labels_pred)
    return score

# calculate the normalized mutual information score for the predicted labels compared to to players labelled by the lol esports region theyve played the most in
def score_regions(labels_pred, uniq_player_ids, data, score_fn):

    player_most_played_region = data.groupby("playerid")["league"].agg(lambda x: x.map(lambda league: league_to_region_dict[league]).mode().iloc[0] if not x.mode().empty else None)
    assert not player_most_played_region.isna().any()

    # make sure the order of the true labels are the same as the order of the predictions
    uniq_player_ids_most_played_region = np.vectorize(lambda x: player_most_played_region[x])(uniq_player_ids)
    _, region_labels = np.unique(uniq_player_ids_most_played_region, return_inverse=True)

    score = score_fn(region_labels, labels_pred)
    return score

# calculate the normalized mutual information score for the predicted labels compared to to players labelled by the continent theyve played the most in
def score_continents(labels_pred, uniq_player_ids, data, score_fn):

    player_most_played_continent = data.groupby("playerid")["league"].agg(lambda x: x.map(lambda league: league_to_continent_dict[league]).mode().iloc[0] if not x.mode().empty else None)
    assert not player_most_played_continent.isna().any()

    # make sure the order of the true labels are the same as the order of the predictions
    uniq_player_ids_most_played_continent = np.vectorize(lambda x: player_most_played_continent[x])(uniq_player_ids)
    _, continent_labels = np.unique(uniq_player_ids_most_played_continent, return_inverse=True)

    score = score_fn(continent_labels, labels_pred)
    return score


from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score # ari
from sklearn.metrics import silhouette_score

def compute_all_metrics(clustering, uniq_player_ids, data):
    score_fn = normalized_mutual_info_score
    nmi_leagues = score_leagues(clustering, uniq_player_ids, data, score_fn)
    nmi_regions = score_regions(clustering, uniq_player_ids, data, score_fn)
    nmi_continents = score_continents(clustering, uniq_player_ids, data, score_fn)
    score_fn = adjusted_rand_score
    ari_leagues = score_leagues(clustering, uniq_player_ids, data, score_fn)
    ari_regions = score_regions(clustering, uniq_player_ids, data, score_fn)
    ari_continents = score_continents(clustering, uniq_player_ids, data, score_fn)
    return {
        "nmi_leagues": nmi_leagues,
        "nmi_regions": nmi_regions,
        "nmi_continents": nmi_continents,
        "ari_leagues": ari_leagues,
        "ari_regions": ari_regions,
        "ari_continents": ari_continents,
    }