import pandas as pd
import numpy as np
import nlu

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from .recommendation_system import RecommendationSystem

df = pd.read_csv("data.csv")
df.columns = ["customer_id", "date", "category"]

ndf = df.drop_duplicates(subset=["customer_id", "category"])
categories = ndf.category.unique()

for category in categories:
    processed_category = [word for word in word_tokenize(category)
                              if (word not in stopwords.words("ukrainian")) and word.isalpha()]
    ndf.loc[ndf["category"] == category, "category"] = ' '.join(processed_category)

lemma_categories = nlu.load('uk.lemma').predict(categories, output_level='document')
lemma_categories.rename({"document": "category"}, axis=1, inplace=True)
ndf = ndf.merge(lemma_categories, on="category")

customers_queries = ndf.set_index(["customer_id", "date"]).drop("category", axis=1)
customers_vs_queries = pd.DataFrame(customers_queries["lem"].tolist(), index=ndf["customer_id"]).stack().reset_index().pivot_table(index="customer_id", columns=0, fill_value=0, aggfunc='size')
customers_vs_queries[customers_vs_queries > 0] = 1

docs = list(pd.Series(ndf.lem.astype(str).unique()).apply(lambda x: x[1:-1].replace("'", "").split(', ')).values)
dictionary = Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

tfidf = TfidfModel(corpus)
tfidf_weights = pd.DataFrame(tfidf[corpus[0]])
for corpora in corpus[1:]:
  tfidf_weights = tfidf_weights.append(pd.DataFrame(tfidf[corpora]), ignore_index=True) 
tfidf_weights.rename({0: "id", 1: "weight"}, axis=1, inplace=True)
tfidf_weights = tfidf_weights.groupby("id").agg({"weight": ["max", "min", "mean"]}).reset_index()
tfidf_weights["id"] = tfidf_weights["id"].apply(lambda x: dictionary[x])

for category, weight in zip(tfidf_weights["id"], tfidf_weights["weight"]["mean"]):
  customers_vs_queries.loc[:, category] = (customers_vs_queries[category] * weight)

user_to_user_sim_matrix = pd.DataFrame(
    cosine_similarity(customers_vs_queries),
    index=customers_vs_queries.index,
    columns=customers_vs_queries.index
)

model = RecommendationSystem(nearest_neighbors_number=3, items_number=10)
model.train(user_to_user_sim_matrix, df)