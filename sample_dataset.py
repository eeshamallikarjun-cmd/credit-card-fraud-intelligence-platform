import pandas as pd

df = pd.read_csv("creditcard.csv")

sample = df.sample(n=10000, random_state=42)

sample.to_csv("creditcard_sample.csv", index=False)

print("Done")