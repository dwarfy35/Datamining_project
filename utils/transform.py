import json
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np


def _is_class(k, p):
    if k in p:
        return 1
    else:
        return 0
    
def _clean(k):
    if "Wukong" in k:
        return "MonkeyKing"
    if "K'Sante" in k:
        return "KSante"
    if "Kai'Sa" in k:
        return "Kaisa"
    if "Renata Glasc" in k:
        return "Renata"
    if "Kog'Maw" in k:
        return "KogMaw"
    if "LeBlanc" in k:
        return "Leblanc"
    if "Nunu & Willump" in k:
        return "Nunu"
    if "Dr. Mundo" in k:
        return "DrMundo"
    if "Rek'Sai" in k:
        return "RekSai"
    else:
        z = k.split("'")
        m = [b.lower() for b in z[1:]]
        z = [z[0]] + m

        return "".join(z).replace(" ", "")
        #return k.replace("'","").replace(" ", "")


# dataframe must have a column called "Champion" with champion names according to oracleelixir
def champion_class_transform(df):
    champion_json_path = Path(__file__).parent / "champion.json"
    with open(champion_json_path, "r", encoding="utf-8") as f:
        champion_json = json.load(f)

    tags_set = set()
    for x in champion_json["data"].keys():
        tags_set.update(set(champion_json["data"][x]["tags"]))
    tags = list(tags_set)
    
    df = df.copy()
    for tag in tags:
        df.loc[:, tag] = df["champion"].apply(lambda k: _is_class(tag, champion_json["data"][_clean(k)]["tags"]))
    
    df = df.drop(columns=["champion"])
    return df

def smart_drop_na(df, column_percentage_threshold=0.2, row_percentage_threshold=0):
    missing_values = df.isna().mean()
    missing_value_columns = missing_values[missing_values > column_percentage_threshold].index
    df = df.drop(columns=missing_value_columns)

    z = df.isna().sum(axis=1)
    # Gameid to drop
    gameids = df[z > len(df.columns) * row_percentage_threshold]["gameid"].unique()
    df = df[~df["gameid"].isin(gameids)]
    return df

from utils.constants import most_relevant_columns, pos_order
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge

time_dependent_cols = ["kills", "deaths", "assists", "doublekills", "damagetochampions", "dpm", "damagetowers", "wardsplaced", "visionscore", "totalgold", "minionkills", "monsterkills"]
time_dependent_cols = [col + f'_{pos}' for col in most_relevant_columns if col != 'gamelength' for pos in pos_order]
simple_scale_model = make_pipeline(SplineTransformer(n_knots=3, degree=3), Ridge(alpha=1))

def scale_df(df, time_dependent_cols=time_dependent_cols, model=simple_scale_model):
    df_residuals = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col not in time_dependent_cols:
            df_residuals[col] = df[col]
            continue
        model.fit(df[['gamelength']], df[col])
        df_residuals[col] = df[col] - model.predict(df[['gamelength']])
    
    return df_residuals

def aggregate_stats(df, id_col):
    df = df[df.groupby(id_col)[id_col].transform('count') > 2]
    
    g = df.groupby(id_col)
    
    res_mean = g.mean().add_suffix('_mean')
    res_var = g.var().add_suffix('_var')
    res_quant = g.quantile([0.25, 0.5, 0.75]).unstack()
    
    res_quant.columns = [f"{c[0]}_q{int(c[1]*100)}" for c in res_quant.columns]
    
    return pd.concat([res_mean, res_var, res_quant], axis=1).reset_index()



random_walk_length = 100
random_walks = []
random_walks_count = 1000
jump_prob = 0.2


def random_walk_embeddings(uniq_player_ids, adj_matrix, random_walk_length=100, random_walks_count=1000, jump_prob=0.2):
    for r in tqdm(range(random_walks_count)):
        
        start = int(len(uniq_player_ids) * np.random.rand())    
        random_walk = [uniq_player_ids[start]]

        for i in range(random_walk_length):
            if np.random.rand() > jump_prob:
                jump = np.random.choice(uniq_player_ids, p=adj_matrix[start]/sum(adj_matrix[start]))
            else:
                jump = uniq_player_ids[int(len(uniq_player_ids) * np.random.rand())]
            random_walk.append(jump)
    #pd.DataFrame(random_walk).value_counts()
        random_walks.append(random_walk)