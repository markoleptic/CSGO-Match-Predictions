import pandas as pd
import numpy as np

df = pd.read_csv('economy.csv',header=0)

print(df.head())

tempdict = []
for index, row in df.iterrows():
    # print(type(row))
    # print(row)
    newrow = row.dropna()
    #print(newrow)
    t1_money = []
    t2_money = []
    win = []
    for label, value in newrow.items():
        if '_t2' in label:
            t1_money.append(value)
        if '_t1' in label:
            t2_money.append(value)
        if 'win' in label:
            win.append(value)
    for i in enumerate(t1_money):
        tempdict.append(dict(t1_money=t1_money[i[0]], t2_money=t2_money[i[0]], winner=win[i[0]]))
    # print(np.array(t1_money))
    # print(np.array(t2_money))
    # print(win)
newdf = pd.DataFrame(tempdict, columns=['t1_money', 't2_money', 'winner'])
#print(newdf)

newdf.to_csv('roundMoneyWinners.csv')
