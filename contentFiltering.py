import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks')
#from autoencoder import AutoEncoder
import matplotlib.pyplot as plt
import pickle
import multiprocessing
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics.pairwise import cosine_similarity

class contentRecommender:
  
  def __init__(self,movies,ratings,tags):
    self.movies=movies
    self.ratings=ratings
    self.tags=tags
    
  def createDocumnet(self):
    print("{} unique movies in tags.csv".format(len(self.tags.movieId.unique())))
    print("the tags data has {} shape".format(self.tags.shape))
    self.ratings = self.ratings.drop_duplicates('movieId')
    print("{} unique movies in ratings.csv".format(len(ratings.movieId.unique())))
    
    self.movies['genres'] = self.movies['genres'].str.replace(pat="|", repl=" ")
    self.movies['genres'] = self.movies['genres'].str.replace(pat="-", repl="")
    self.tags.fillna("", inplace=True)
    self.tags = pd.DataFrame(self.tags.groupby('movieId')['tag'].apply(lambda x: "{%s}" % ' '.join(x)))
    self.tags.reset_index(inplace=True)
    movie_id = self.tags.movieId
    print("There are {} unique movies".format(len(movie_id)))
    self.tags = pd.merge(self.movies, self.tags,on='movieId', how='right')
    self.tags['document'] = self.tags[[ 'genres','tag']].apply(lambda x: ' '.join(x), axis=1)
  
  def get_movie_id(self,title):
    return self.tags[self.tags['title'].str.contains(title)]
  
  def getEmbeddings(self):
    if(os.path.exists('/content/drive/My Drive/autoencoder_embeddings.pkl')):
      self.content_embeddings = pd.read_pickle( "/content/drive/My Drive/autoencoder_embeddings.pkl")
      self.content_embeddings = pd.DataFrame(content_embeddings)
    else:
      tfidf=TfidfVectorizer(ngram_range=(0,1),min_df=.0001,stop_words='english')
      tfidf_matrix = tfidf.fit_transform(self.tags['document'])
      tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=self.tags.index.tolist())
      ae = AutoEncoder(tfidf_df, validation_perc=0.1, lr=1e-3, intermediate_size=5000, encoded_size=100)
      ae.train_loop(epochs=30)
      with open('/content/drive/My Drive/autoencoder.pkl', 'wb') as fh:
        pickle.dump(ae, fh)
      losses = pd.DataFrame(data=list(zip(ae.train_losses, ae.val_losses)), columns=['train_loss', 'validation_loss'])
      losses['epoch'] = (losses.index + 1) / 3
      encoded = ae.get_encoded_representations()
      self.content_embeddings=encoded
      
      with open('/content/drive/My Drive/autoencoder_embeddings.pkl', 'wb') as fh:
        pickle.dump(encoded, fh)
    return self.content_embeddings
  
  def recommend(self,movie_id,n):
    self.ids=self.content_embeddings.index.tolist()
    similarity_matrix = pd.DataFrame(cosine_similarity(X=self.content_embeddings),index=self.ids)
    similarity_matrix.columns = self.ids
    similar_items = pd.DataFrame(similarity_matrix.loc[movie_id])
    similar_items.columns = ["similarity_score"]
    similar_items = similar_items.sort_values('similarity_score', ascending=False)
    similar_items = similar_items.head(n)
    similar_items.reset_index(inplace=True)
    similar_items = similar_items.rename(index=str, columns={"index": "item_id"})
    similar_items = similar_items.to_dict()
    similar_movies = pd.DataFrame(similar_items)
    similar_movies.set_index('item_id', inplace=True)
    sim_df = pd.merge(movies, similar_movies, left_index=True, right_index=True)
    sim_df.sort_values('similarity_score', ascending=False, inplace=True)
    return sim_df.head(n)
  