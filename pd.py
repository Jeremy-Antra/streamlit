import pandas as pd

df = pd.read_csv('E:\\10yearsdatawithstock.csv')

df = df.drop(df[df['CompanyId']>9000].index)

df.to_csv('10yearsdatawithstock.csv')