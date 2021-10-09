import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('final.csv')

# Converting the title to lower case
for i in df["title"]:
    df["title"].replace({i:i.lower()},inplace=True)

# print(df["title"])

cv = CountVectorizer(stop_words='english')
data = cv.fit_transform(df['title'])

cos = cosine_similarity(data,data)
# print(cos)

# Resetting the index of the dataframe to the title of the articles
df = df.reset_index()
indices = pd.Series(df.index,index=df['title'])

def get_recommendation(title,classifier):
    idx = indices[title]
    sim_scores = list(enumerate(classifier[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# print(get_recommendation("Don't document your code. Code your documentation.".lower(), cos))

# print(get_recommendation("Ethereum, a Virtual Currency, Enables Transactions That Rival Bitcoin's".lower(), cos))

