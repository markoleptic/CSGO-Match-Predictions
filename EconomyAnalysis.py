import pandas as pd
import numpy as np

df = pd.read_csv("economy.csv",header=0,index_col=False,low_memory=False)
results_df = pd.read_csv("results.csv",header=0,index_col=False,low_memory=False)
maps = [
    "Nuke",
    "Dust2",
    "Mirage",
    "Inferno",
    "Train",
    "Default",
    "Vertigo",
    "Overpass",
    "Cobblestone",
    "Cache",
]
# print(df.head())
# print(results_df.head())

tempdict = []
for index, row in df.iterrows():
    newrow = row.dropna()
    matchingRows = results_df.loc[results_df["match_id"] == newrow["match_id"]]
    matchingRows = matchingRows.loc[results_df["_map"] == newrow["_map"]]
    if matchingRows.empty:
        continue

    team_1 = newrow["team_1"]
    team_2 = newrow["team_2"]
    rank_1 = matchingRows["rank_1"].values[0]
    rank_2 = matchingRows["rank_2"].values[0]
    t1_start_side = newrow["t1_start"]
    t2_start_side = newrow["t2_start"]
    if matchingRows["map_winner"].values[0] == 1:
        match_winner = team_1
    else:
        match_winner = team_2

    if matchingRows.shape[0] != 1:
        print("not 1")
        print(matchingRows)
        exit()

    t1_money = []
    t2_money = []
    round_winner = []
    # get the money for each team and winner for each round in the match
    for label, value in newrow.items():
        if "_t1" in label:
            t1_money.append(value)
        if "_t2" in label:
            t2_money.append(value)
        if "win" in label:
            round_winner.append(value)

    num_rounds = int(newrow.last_valid_index()[:2])
    # create as many rows in data table as there are rounds
    for round in range(num_rounds):
        if (round < 15):
            t1_side = t1_start_side
            t2_side = t2_start_side
        else:
            t1_side = t2_start_side
            t2_side = t1_start_side
        tempdict.append(
            dict(
                round_number = round + 1,
                team_1 = team_1,
                team_2 = team_2,
                t1_side = t1_side,
                t2_side = t2_side,
                t1_money = t1_money[round],
                t2_money = t2_money[round],
                t1_rank = rank_1,
                t2_rank = rank_2,
                map = maps.index(newrow["_map"]),
                winner = int(round_winner[round]),
                match_winner = match_winner,
            )
        )
newdf = pd.DataFrame(
    tempdict,
    columns=[
        "round_number",
        "team_1",
        "team_2",
        "t1_side",
        "t2_side",
        "t1_money",
        "t2_money",
        "t1_rank",
        "t2_rank",
        "map",
        "winner",
        "match_winner",
    ],
)
newdf.to_csv("roundMoneyWinners2.csv", index=False)
