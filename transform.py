import json
import pandas as pd
from pathlib import Path


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
def champion_class_transform(dataframe):
    cwd = Path('.')
    champion_json_path = cwd / "champion.json"
    with open(champion_json_path, "r", encoding="utf-8") as f:
        champion_json = json.load(f)

    tags_set = set()
    for x in champion_json["data"].keys():
        tags_set.update(set(champion_json["data"][x]["tags"]))
    tags = list(tags_set)
    
    df = dataframe.copy()
    for tag in tags:
        df.loc[:, tag] = df["champion"].apply(lambda k: _is_class(tag, champion_json["data"][_clean(k)]["tags"]))
    
    df = df.drop(columns=["champion"])
    return df