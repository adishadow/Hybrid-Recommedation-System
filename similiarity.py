import numpy as np
import pandas as pd
import os
import sys
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import matplotlib.pyplot as plt
from importlib import reload
import pandas as pd
from  keras.models import load_model
from keras import backend as K


movies=pd.read_csv('C:\\Users\\adity\\Desktop\\New folder (3)\\dataset\\mov.csv')

def getEnsembleRecommendation(userid,movieid):
  cwd = os.getcwd()
  content_embeddings = pd.read_pickle("C:\\Users\\adity\\Desktop\\New folder (3)\\script\\compiled model\\autoencoder_embeddings.pkl")
  content_embeddings = pd.DataFrame(content_embeddings)
  print(content_embeddings.shape)
  content_embeddings.head()
  ids=content_embeddings.index.tolist()
  movie_data = pd.read_pickle("C:\\Users\\adity\\Desktop\\New folder (3)\\script\\compiled model\\movie_data.pkl")
  
  if(os.path.exists('C:\\Users\\adity\\Desktop\\New folder (3)\\script\\compiled model\\collaborative.h5')):
      model=load_model('C:\\Users\\adity\\Desktop\\New folder (3)\\script\\compiled model\\collaborative.h5')
      movie_em=model.get_layer('Movie-Embedding')
  
  similarity_matrix = pd.DataFrame(cosine_similarity(X=content_embeddings),index=ids)
  similarity_matrix.columns = ids                    
  mid=movies[movies['movieId']==movieid].index.tolist()
  similar_items = pd.DataFrame(similarity_matrix.iloc[mid[0]])
  similar_items.columns = ["similarity_score"]
  similar_items = similar_items.sort_values('similarity_score', ascending=False)
  similar_items = similar_items.head(26744)
  similar_items.reset_index(inplace=True)
  similar_items = similar_items.rename(index=str, columns={"index": "item_id"})
  similar_items = similar_items.to_dict()
  similar_movies = pd.DataFrame(similar_items)
  similar_movies.set_index('item_id', inplace=True)
  sim_df = pd.merge(movies, similar_movies, left_index=True, right_index=True)
  sim_df.sort_values('similarity_score', ascending=False, inplace=True)
  sim_df_cont = sim_df.rename(index=str, columns={"similarity_score": "collaborative_similarity_score"})
  
  
  movie_dataDf = pd.DataFrame(movie_data,columns=['movieId'])
  user = np.array([userid for i in range(len(movie_data))])
  prediction=model.predict([user,movie_data])
  prediction = pd.DataFrame(prediction,columns=['sim_score'])
  prediction = pd.concat([movie_dataDf,prediction],axis=1)
  #self.prediction=np.array([a[0] for a in self.prediction])
  #self.recommended_book_ids = (-self.prediction).argsort()[:15]
  prediction.sort_values('sim_score',ascending=False,inplace=True)
  
  sim_df_avg = pd.merge(sim_df_cont, pd.DataFrame(prediction[['movieId','sim_score']]), on="movieId")
  sim_df_avg['average_similarity_score'] = (sim_df_avg['collaborative_similarity_score'] + sim_df_avg['sim_score'])/2
  #sim_df_avg.drop("collaborative_similarity_score", axis=1, inplace=True)
  #sim_df_avg.drop("content_similarity_score", axis=1, inplace=True)
  sim_df_avg.sort_values('average_similarity_score', ascending=False, inplace=True)
    
  #save recs locally
  #sim_df_avg.head(20).to_csv(file_path, index=False, header=True)
  return sim_df_avg.head(20)

  
  
  
  
  