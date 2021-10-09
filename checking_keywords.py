import pandas as pd

df = pd.read_csv("final_dataset.csv")
print(df.shape)

print(df["keywords"].tolist()[0])


