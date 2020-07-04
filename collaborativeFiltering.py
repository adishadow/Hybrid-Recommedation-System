import numpy as np
import pandas as pd
from keras.models import Model
import matplotlib as plt
import seaborn as sbn
import os
import sys
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from sklearn.model_selection import train_test_split
from  keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import keras.regularizers
import pickle


class collab:
  def __init__(self,ratings,movies):
    self.ratings=ratings
    self.movies=movies
    self.n_movies=ratings.movieId.nunique()
    self.n_users  = ratings.userId.nunique()
    self.mov_rati_join = pd.merge(left=ratings,right=movies,on='movieId',how='left')
    
    
  def intialize_model(self):
    movie_input=Input(shape=[1],name='Movie-Input')
    movie_embedding=Embedding(self.n_movies+1, 5 ,name='Movie-Embedding')(movie_input)
    movie_flatten=Flatten(name='Flatten-Movie')(movie_embedding)
    #user embbeding
    user_input=Input(shape=[1],name='User-Input')
    user_embedding=Embedding(self.n_users+1, 5 ,name='User-Embedding')(user_input)
    user_flatten=Flatten(name='Flatten-User')(user_embedding)
    
    conc = Concatenate()([movie_flatten, user_flatten])
    
    fc1 = Dense(200, activation='relu',kernel_initializer='random_uniform',kernel_regularizer=keras.regularizers.l2(0.01))(conc)
    fc2 = Dense(100, activation='relu',kernel_initializer='random_uniform',kernel_regularizer=keras.regularizers.l2(0.01))(fc1) 
    fc3 = Dense(50,activation='relu',kernel_initializer='random_uniform',kernel_regularizer=keras.regularizers.l2(0.01))(fc2)
    out = Dense(1)(fc3)
    
    self.model = Model([movie_input,user_input],out)
    self.model.compile('adam' ,'mean_squared_error',metrics=['accuracy'])
    
    return self
  
  def fit(self):
    self.train,self.test= train_test_split(self.ratings,test_size=.15,random_state=42)
    if(os.path.exists('/content/drive/My Drive/collaborative.h5')):
      self.model=load_model('/content/drive/My Drive/collaborative.h5')
      movie_em=self.model.get_layer('Movie-Embedding')
    else:
      history=self.model.fit([self.train.movieId,self.train.userId],self.train.rating,epochs=10,verbose=1,batch_size=150,)
      self.model.save('/content/drive/My Drive/collaborative.h5')
      with open('/content/drive/My Drive/collab_embed.pkl', 'wb') as fh:
        pickle.dump(history, fh)
      movie_em=model.get_layer('Movie-Embedding')
      
  def evaluate(self):
    self.model.evaluate([self.test.movieId,self.test.userId],self.test.rating)
    
  def getRecommendation(self,userId):
    movie_data=np.array(list(set(ratings.movieId.unique())))
    movie_dataDf = pd.DataFrame(movie_data,columns=['movieId'])
    user = np.array([userId for i in range(len(movie_data))])
    self.prediction=self.model.predict([user,movie_data])
    self.prediction = pd.DataFrame(self.prediction,columns=['sim_score'])
    self.prediction = pd.concat([movie_dataDf,self.prediction],axis=1)
    #self.prediction=np.array([a[0] for a in self.prediction])
    #self.recommended_book_ids = (-self.prediction).argsort()[:15]
    self.prediction.sort_values('sim_score',ascending=False,inplace=True)
    self.recommended_book_ids=self.prediction.head(12).movieId
    print(movies[movies['movieId'].isin(self.recommended_book_ids)])
    return self.recommended_book_ids,movies[movies['movieId'].isin(self.recommended_book_ids)]

                                  
  
    