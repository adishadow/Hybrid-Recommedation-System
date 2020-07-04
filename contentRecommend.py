import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys

def recommend(movie_id,n):
    content_embeddings = pd.read_pickle( "C:\\Users\\adity\\Desktop\\New folder (3)\\script\\compiled model\\autoencoder_embeddings.pkl")
    content_embeddings = pd.DataFrame(content_embeddings)
    ids=content_embeddings.index.tolist()
    similarity_matrix = pd.DataFrame(cosine_similarity(X=content_embeddings),index=ids)
    similarity_matrix.columns = ids
    similar_items = pd.DataFrame(similarity_matrix.loc[movie_id])
    similar_items.columns = ["similarity_score"]
    similar_items = similar_items.sort_values('similarity_score', ascending=False)
    similar_items = similar_items.head(n)
    similar_items.reset_index(inplace=True)
    similar_items = similar_items.rename(index=str, columns={"index": "item_id"})
    similar_items = similar_items.to_dict()
    similar_movies = pd.DataFrame(similar_items)
    similar_movies.set_index('item_id', inplace=True)
    movies=pd.read_csv("C:\\Users\\adity\\Desktop\\New folder (3)\\dataset\\mov.csv")
    sim_df = pd.merge(movies, similar_movies, left_index=True, right_index=True)
    sim_df.sort_values('similarity_score', ascending=False, inplace=True)
    return sim_df.head(n)