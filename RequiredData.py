import pandas as pd

df = pd.read_csv('economy.csv',usecols=['team_1','team_2','_map','1_t1','1_t2'])
df.to_csv("RequiredData.csv")