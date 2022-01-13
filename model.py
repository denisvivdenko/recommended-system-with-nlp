import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import nlu
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from recommendation_system import RecommendationSystem

print("###READ DATA\n")
df = pd.read_csv("data.csv")
df.columns = ["customer_id", "date", "category"]
df["category"] = df["category"].str.lower()
print(df.head())

print("###DROP DUPLICATES\n")
ndf = df.drop_duplicates(subset=["customer_id", "category"])
categories = ndf.category.unique()
print(ndf.head())

print("###REMOVE STOPWORDS\n")
for category in tqdm(categories):
    processed_category = [word for word in word_tokenize(category)
                              if (word not in stopwords.words("ukrainian")) and word.isalpha()]
    ndf.loc[ndf["category"] == category, "category"] = ' '.join(processed_category)
print(ndf.head())

print("\n###LEMMATIZATION\n")
lemma_categories = nlu.load('uk.lemma').predict(categories, output_level='document')
lemma_categories.rename({"document": "category"}, axis=1, inplace=True)
ndf = ndf.merge(lemma_categories, on="category")
print(ndf.head(), "\n")

print("\n###VECTORIZE CATEGORIES NAMES\n")
customers_queries = ndf.set_index(["customer_id", "date"]).drop("category", axis=1)
customers_vs_queries = pd.DataFrame(customers_queries["lem"].tolist(), index=ndf["customer_id"]).stack().reset_index().pivot_table(index="customer_id", columns=0, fill_value=0, aggfunc='size')
customers_vs_queries[customers_vs_queries > 0] = 1
print(customers_vs_queries.head())

print("\n###CREATE DICTIONARY AND CORPUS\n")
docs = list(pd.Series(ndf.lem.astype(str).unique()).apply(lambda x: x[1:-1].replace("'", "").split(', ')).values)
dictionary = Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

print("###TF-IDF\n")
tfidf = TfidfModel(corpus)
tfidf_weights = pd.DataFrame(tfidf[corpus[0]])
for corpora in corpus[1:]:
  tfidf_weights = tfidf_weights.append(pd.DataFrame(tfidf[corpora]), ignore_index=True) 
tfidf_weights.rename({0: "id", 1: "weight"}, axis=1, inplace=True)
tfidf_weights = tfidf_weights.groupby("id").agg({"weight": ["max", "min", "mean"]}).reset_index()
tfidf_weights["id"] = tfidf_weights["id"].apply(lambda x: dictionary[x])
print(tfidf_weights.head())

print("\n###WEIGHT VECTORIZED WORDS\n")
for category, weight in tqdm(zip(tfidf_weights["id"], tfidf_weights["weight"]["mean"])):
  customers_vs_queries.loc[:, category] = (customers_vs_queries[category] * weight)

print("###FACTORIZE MATRIX\n")
user_to_user_sim_matrix = pd.DataFrame(
    cosine_similarity(customers_vs_queries),
    index=customers_vs_queries.index,
    columns=customers_vs_queries.index
)
print(user_to_user_sim_matrix.head())

print("###TO SPARSE MATRIX\n")
sparse_factorized_matrix = csr_matrix(user_to_user_sim_matrix)
print(type(sparse_factorized_matrix))

print("###MODEL\n")
model = RecommendationSystem(nearest_neighbors_number=3, items_number=10)
model.train(sparse_factorized_matrix, user_to_user_sim_matrix.columns, df)
print(model.predict(50081964))

print("###SAVING MODEL TO model.pkl")
pickle.dump(model, open('model.pkl','wb'))