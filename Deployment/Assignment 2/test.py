import pandas as pd

df = pd.read_csv(r'C:\Users\kiran\Downloads\Yelp JSON\review.csv')

print(df)

print(df.loc[0])
print(df.loc[0,'text'])