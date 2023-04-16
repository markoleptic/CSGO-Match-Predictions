import pandas as pd
import numpy as np

df = pd.read_csv('economy.csv', header=0, index_col=False)
results_df = pd.read_csv('results.csv',header=0, index_col=False)
maps = ['Nuke', 'Dust2', 'Mirage', 'Inferno', 'Train', 'Default', 'Vertigo', 'Overpass', 'Cobblestone', 'Cache']
print(df.head())
print(results_df.head())
tempdict = []
for index, row in df.iterrows():
    newrow = row.dropna()
    t1_money = []
    t2_money = []
    win = []
    matchingRows = results_df.loc[results_df['match_id'] == newrow['match_id']]
    if matchingRows.empty:
        continue
    for label, value in newrow.items():
        if '_t2' in label:
            t1_money.append(value)
        if '_t1' in label:
            t2_money.append(value)
        if 'win' in label:
            win.append(value)
    for i in enumerate(t1_money):
        tempdict.append(dict(t1_money=t1_money[i[0]], t2_money=t2_money[i[0]], t1_rank=matchingRows['rank_1'].values[0], t2_rank=matchingRows['rank_2'].values[0], map=maps.index(newrow['_map']), winner=int(win[i[0]])))
newdf = pd.DataFrame(tempdict, columns=['t1_money', 't2_money', 't1_rank', 't2_rank', 'map', 'winner'])
newdf.to_csv('roundMoneyWinners.csv', index=False)
