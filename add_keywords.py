from sklearn.feature_extraction.text import CountVectorizer
import csv,sys
import pandas as pd

df = pd.read_csv('final.csv')

print(df.shape)


# Storing the keywords of all the titles in an array
keywords = []

for title in df['title']:
    model = CountVectorizer(stop_words='english')
    temp = []
    temp.append(title)
    try:
        value = model.fit(temp)
        keywords.append(value.vocabulary_)
    except ValueError:
        keywords.append("")



df['keywords'] = keywords

df.to_csv('final_dataset.csv',index=False)